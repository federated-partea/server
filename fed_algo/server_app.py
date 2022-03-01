import logging
from time import sleep

import numpy as np
import pandas as pd
from django.utils import timezone
from lifelines.utils import StepSizer

import fed_algo.survival_analysis.cox_ph_regression as coxph
from fed_algo.models import Project
from fed_algo.storage import delete_blob, cleanup
from fed_algo.survival_analysis.plot import plot_cox_plotly, plot_km_plotly, plot_na_plotly
from fed_algo.survival_analysis.univariate_analysis import univariate_analysis, pairwise_logrank_test
from pet.smpc_agg import distribute_smpc, aggregate_smpc

logger = logging.getLogger(__name__)


def prepare_univarate_results(results, aggregation, project):
    plot = None
    logger.info(f'[{project.id}] Prepare univariate results')
    if len(aggregation.keys()) >= 2:
        p_values = pairwise_logrank_test(aggregation)
        project.create_result_table(table_name=str("Pairwise Logrank Test"),
                                    columns=p_values.reset_index().astype(str).columns,
                                    plot=plot_km_plotly(results["KM"], project.conditions))
        i = 0
        table = p_values.drop_duplicates()
        for row in table.itertuples():
            row_list = [str(row.Index[0]), str(row.Index[1])]
            values = [str(x) for x in list(row[1:])]
            row_list.extend(values)
            project.create_result_row(table_name=str("Pairwise Logrank Test"), row_name=str(i), values=row_list)
            i += 1
    for result in results.keys():
        try:
            if result == "KM":
                plot = plot_km_plotly(data=results[result])
            elif result == "NA":
                plot = plot_na_plotly(data=results[result])
        except Exception as e:
            logger.error(f'Error while creating plots: {e}')

        project.create_result_table(table_name=str(result),
                                    columns=results[result].reset_index().rename(
                                        {"index": "timeline"}, axis=1).astype(str).columns,
                                    plot=plot)
        i = 0
        table = results[result].drop_duplicates()
        for row in table.itertuples():
            row_list = [str(x) for x in list(row)]
            project.create_result_row(table_name=str(result), row_name=str(i), values=row_list)
            i += 1


def prepare_cox_results(summary_df: pd.DataFrame, params, standard_errors, global_c, sample_number: int,
                        project: Project):
    logger.info(f'[{project.id}] Prepare cox results')

    plot = plot_cox_plotly(params, standard_errors, alpha=0.05)

    summary_df = summary_df.reset_index().rename({"index": "covariate"}, axis=1)
    project.sample_number = int(sample_number)
    project.c_index = str(global_c)
    project.create_result_table(table_name=str("COX"),
                                columns=summary_df.astype(str).columns,
                                plot=plot)
    i = 0
    for row in summary_df.itertuples():
        row_list = [str(x) for x in list(row)]
        project.create_result_row(table_name=str("COX"), row_name=str(i), values=row_list[1:])
        i += 1


class ServerApp:
    def __init__(self, project: Project, ready_func, old_aggregate, old_data, memory):

        self.project = project
        self.ready_func = ready_func
        self.old_aggregate = old_aggregate
        self.old_data = old_data
        self.memory = memory

    def initialize(self):
        self.ready_func(False, agg=self.old_aggregate, data=self.old_data, memory=self.memory)

    def aggregate_univariate(self, client_data):
        if self.project.internal_state == "init":
            logger.info(f'[{self.project.id}] Univariate aggregation')
            results, aggregation, n_samples = univariate_analysis(client_data, int(self.project.privacy_level))
            prepare_univarate_results(results, aggregation, self.project)
            sleep(1)
            self.project.internal_state = "finished"
            self.ready_func(True, {}, client_data, self.memory)
            self.project.sample_number = n_samples
            self.project.state = "finished"
            cleanup(self.project)
            self.project.run_end = timezone.now()

    def smpc_aggregate_univariate(self, client_data):
        if self.project.internal_state == 'init':
            send_to_local = client_data
            self.project.internal_state = "local_calc"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == 'local_calc':
            data_to_broadcast = distribute_smpc(client_data)
            send_to_local = data_to_broadcast
            self.project.internal_state = "smpc_agg"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == 'smpc_agg':
            data = list(client_data.values())
            global_agg = aggregate_smpc(data, exp=0)
            dfs = {}

            global_agg = global_agg["local_results"]
            for key in global_agg:
                df = pd.DataFrame.from_dict(global_agg[key])
                df.index = df.index
                dfs[key] = df
            results, aggregation_dfs = univariate_analysis(dfs, privacy_level=self.project.privacy_level, smpc=True)
            n_samples = 0
            for key in aggregation_dfs:
                n_samples = n_samples + aggregation_dfs[key].iloc[0, 1]

            prepare_univarate_results(results, aggregation_dfs, self.project)
            sleep(1)
            self.project.internal_state = "finished"
            self.ready_func(True, {"state": "finished"}, None)
            self.project.sample_number = n_samples
            self.project.state = "finished"
            cleanup(self.project)
            self.project.run_end = timezone.now()

    def aggregate_cox(self, client_data):
        if self.project.internal_state == "init":
            logger.info(f'[{self.project.id}] Compute global mean')
            norm_mean = coxph.compute_global_mean(client_data)
            self.project.internal_state = "norm_std"
            send_to_local = {"norm_mean": norm_mean}
            self.memory = send_to_local.copy()
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "norm_std":
            logger.info(f'[{self.project.id}] Compute global std')
            norm_std = coxph.compute_global_std(client_data)
            self.project.internal_state = "local_init"
            send_to_local = {"norm_std": norm_std}
            self.memory.update(send_to_local)
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "local_init":
            logger.info(f'[{self.project.id}] Initialization')
            D, zr, count_d, n_samples = coxph.global_initialization(client_data)
            covariates = zr.axes[0]
            beta = np.zeros((len(covariates),))
            step_sizer = StepSizer(None)
            step_size = step_sizer.next()
            delta = np.zeros_like(beta)
            iteration = 0
            converging = True

            self.memory.update({"beta": beta, "iteration": iteration, "D": D, "zr": zr,
                                "count_d": count_d, "n_samples": n_samples, "step_sizer": step_sizer,
                                "step_size": step_size, "delta": delta, "converging": converging})
            send_to_local = {"beta": beta, "iteration": iteration}
            self.project.internal_state = "iteration_update"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "iteration_update":
            iteration = self.memory["iteration"]
            logger.info(f'[{self.project.id}] Iteration update {str(iteration)}')
            beta = self.memory["beta"]
            zr = self.memory["zr"]
            step_sizer = self.memory["step_sizer"]
            step_size = self.memory["step_size"]
            iteration = self.memory["iteration"]
            n_samples = self.memory["n_samples"]
            count_d = self.memory["count_d"]
            D = self.memory["D"]
            converging = self.memory["converging"]
            new_beta, converging, hessian, step_size, iteration, delta = coxph.iteration_update(
                client_data, beta, zr, converging, step_sizer, step_size, iteration,
                n_samples, count_d, D, penalization=self.project.penalizer,
                l1_ratio=self.project.l1_ratio, max_steps=self.project.max_iters)

            if converging:
                send_to_local = {"beta": new_beta, "iteration": iteration}
                self.memory.update(send_to_local)
                self.project.internal_state = "iteration_update"
                self.ready_func(False, send_to_local, client_data, self.memory)
            else:
                logger.info(f'[{self.project.id}] Stopping criterion fulfilled')
                norm_std = self.memory["norm_std"]
                summary_df, params, standard_errors = coxph.create_summary(norm_std, new_beta, zr, hessian)
                send_to_local = {"summary_df": summary_df, "params": params, "standard_errors": standard_errors,
                                 "beta": new_beta}
                self.memory.update(send_to_local)
                self.project.internal_state = "c-index"
                self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "c-index":
            logger.info(f'[{self.project.id}] Finishing...')
            global_c, sample_number = coxph.calculate_concordance_index(client_data)
            prepare_cox_results(self.memory["summary_df"], self.memory["params"],
                                self.memory["standard_errors"], global_c, sample_number, self.project)
            sleep(1)
            self.project.internal_state = "finished"
            self.ready_func(True, {"state": "finished"}, None, self.memory)
            self.project.state = "finished"
            logger.info(f'[{self.project.id}] Computation finished')
            cleanup(self.project)
            self.project.run_end = timezone.now()

    def smpc_aggregate_cox(self, client_data):
        exp = 10
        if self.project.internal_state == 'init':
            send_to_local = client_data
            self.project.internal_state = "norm_mean"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "norm_mean":
            data_to_broadcast = distribute_smpc(client_data)
            send_to_local = data_to_broadcast
            self.project.internal_state = "smpc_agg_norm_mean"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "smpc_agg_norm_mean":
            data = list(client_data.values())
            global_agg = aggregate_smpc(data, exp)
            n_samples = global_agg["n_samples"] * 10**exp
            self.memory = {"n_samples": n_samples}
            global_mean = pd.Series(global_agg["mean"]) / n_samples
            send_to_local = global_mean
            self.project.internal_state = "norm_std"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "norm_std":
            data_to_broadcast = distribute_smpc(client_data)
            send_to_local = data_to_broadcast
            self.project.internal_state = "smpc_agg_norm_std"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "smpc_agg_norm_std":
            data = list(client_data.values())
            global_agg = aggregate_smpc(data, exp)
            n_samples = self.memory["n_samples"]
            norm_std = np.sqrt(pd.Series(global_agg["std"]) / (n_samples - 1))
            self.memory.update({"norm_std": norm_std})
            send_to_local = norm_std
            self.project.internal_state = "local_init"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "local_init":
            data_to_broadcast = distribute_smpc(client_data)
            send_to_local = data_to_broadcast
            self.project.internal_state = "smpc_agg_local_init"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "smpc_agg_local_init":
            data = list(client_data.values())
            global_agg = aggregate_smpc(data, exp)
            global_zlr = pd.Series(global_agg["zlr"])
            covariates = pd.Series(global_zlr).index.tolist()
            beta = np.zeros((len(covariates),))
            step_sizer = StepSizer(None)
            step_size = step_sizer.next()
            delta = np.zeros_like(beta)
            beta += step_size * delta
            global_count_d = pd.Series(global_agg["numb_d_set"])
            global_count_d.index = global_count_d.index.astype(float)
            global_count_d.index = global_count_d.index.astype(str)
            iteration = 0
            converging = True
            global_distinct_times = np.arange(self.project.from_time, self.project.to_time,
                                              self.project.step_size).tolist()
            global_distinct_times.reverse()
            self.memory.update(
                {"global_zlr": global_zlr, "covariates": covariates, "beta": beta, "step_sizer": step_sizer,
                 "step_size": step_size, "global_count_d": global_count_d, "iteration": iteration,
                 "converging": converging, "global_distinct_times": global_distinct_times})
            send_to_local = [beta, iteration]
            self.project.internal_state = "iteration_update"
            self.ready_func(False, send_to_local, data, self.memory)
        elif self.project.internal_state == "iteration_update":
            data_to_broadcast = distribute_smpc(client_data)
            send_to_local = data_to_broadcast
            self.project.internal_state = "smpc_agg_update"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "smpc_agg_update":
            data = list(client_data.values())
            global_agg = aggregate_smpc(data, exp)
            iteration = self.memory["iteration"]
            logger.info(f'[{self.project.id}] Iteration update {str(iteration)}')
            beta = self.memory["beta"]
            zr = self.memory["global_zlr"]
            step_sizer = self.memory["step_sizer"]
            step_size = self.memory["step_size"]
            iteration = self.memory["iteration"]
            n_samples = self.memory["n_samples"]
            count_d = self.memory["global_count_d"]
            D = self.memory["global_distinct_times"]
            converging = self.memory["converging"]
            new_beta, converging, hessian, step_size, iteration, delta = coxph.iteration_update(
                global_agg, beta, zr, converging, step_sizer, step_size, iteration,
                n_samples, count_d, D, penalization=self.project.penalizer,
                l1_ratio=self.project.l1_ratio, max_steps=self.project.max_iters, smpc=self.project.smpc)

            if converging:
                send_to_local = [new_beta, iteration]
                self.memory.update({"beta": new_beta, "iteration": iteration})
                self.project.internal_state = "iteration_update"
                self.ready_func(False, send_to_local, client_data, self.memory)
            else:
                logger.info(f'[{self.project.id}] Stopping criterion fulfilled')
                norm_std = self.memory["norm_std"]
                summary_df, params, standard_errors = coxph.create_summary(norm_std, new_beta, zr, hessian)
                send_to_local = params
                self.memory.update({"summary_df": summary_df, "params": params, "standard_errors": standard_errors})
                self.project.internal_state = "c_index"
                self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "c_index":
            data_to_broadcast = distribute_smpc(client_data)
            send_to_local = data_to_broadcast
            self.project.internal_state = "smpc_agg_c_index"
            self.ready_func(False, send_to_local, client_data, self.memory)
        elif self.project.internal_state == "smpc_agg_c_index":
            data = list(client_data.values())
            global_agg = aggregate_smpc(data, 0)
            n_samples = self.memory["n_samples"]
            global_c = (global_agg["c-index"] / 10) / n_samples
            summary_df = self.memory["summary_df"]
            params = self.memory["params"]
            standard_errors = self.memory["standard_errors"]
            prepare_cox_results(summary_df, params, standard_errors, global_c, n_samples, self.project)
            self.project.internal_state = "finished"
            self.ready_func(True, {"finished": True}, client_data, self.memory)
            logger.info(f'[{self.project.id}] Computation finished')
            self.project.run_end = timezone.now()
            delete_blob(f'p{self.project.id}_memory')
            sleep(2)
            cleanup(self.project)
            self.project.state = "finished"

    def aggregate(self, client_data):
        try:
            self.check_error(client_data)
            logger.info(f'[{self.project.id}] Aggregate')
            logger.info(f'[{self.project.id}] {self.project.internal_state}')
            if self.project.method == "univariate":
                if not self.project.smpc:
                    self.aggregate_univariate(client_data)
                else:
                    self.smpc_aggregate_univariate(client_data)
            elif self.project.method == "cox":
                if not self.project.smpc:
                    self.aggregate_cox(client_data)
                else:
                    self.smpc_aggregate_cox(client_data)
            self.project.save()

        except Exception as e:
            self.project.state = "error"
            self.project.error_message = f'Error: {e}'
            self.project.run_end = timezone.now()
            self.project.save()
            logger.warning(f'[{self.project.id}] Computation finished with an error: {e}')
            send_to_local = {"error": self.project.error_message}
            self.ready_func(True, send_to_local, client_data, self.memory)
            delete_blob(f'p{self.project.id}_memory')
            cleanup(self.project)
            raise

    def check_error(self, client_data):
        for client_id in client_data.keys():
            client = client_data[client_id]
            try:
                if "error" in client.keys():
                    self.project.state = "error"
                    self.project.error_message = client["error"]
                    self.project.run_end = timezone.now()
                    self.project.save()
                    logger.warning(f'[{self.project.id}] Computation finished with a client side error.')
                    self.ready_func(True, {"error": self.project.error_message}, client_data, self.memory)
                    delete_blob(f'p{self.project.id}_memory')
                    cleanup(self.project)
                    return
            except AttributeError:
                pass
