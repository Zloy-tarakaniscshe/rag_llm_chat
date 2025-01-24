"""
ASGI config for chat_ai project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
import django

from channels.routing import get_default_application
# from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chat_ai.settings')

# application = get_asgi_application()
application = get_default_application()
django.setup()
