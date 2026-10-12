"""Smart Agent Web application package.

Import the ASGI application from ``smart_agent.web.app`` explicitly. Keeping
this package initializer free of an ``app`` attribute prevents Python's
submodule import resolution from shadowing ``smart_agent.web.app``.
"""
