def jwt_token_middleware(get_response):
    # One-time configuration and initialization.

    def middleware(request):

        # Code to be executed for each request before
        # the view (and later middleware) are called.

        if not request.META.get('HTTP_AUTHORIZATION'):
            cookie_access_token = request.COOKIES.get("access_token")
            query_access_token = request.GET.get("access_token")

            access_token = query_access_token if query_access_token else cookie_access_token

            if access_token:
                request.META['HTTP_AUTHORIZATION'] = f"Bearer {access_token}"

        response = get_response(request)

        # Code to be executed for each request/response after
        # the view is called.

        return response

    return middleware
