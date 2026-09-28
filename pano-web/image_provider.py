"""Explicit image provider selection; importing this module makes no requests."""
import importlib
from project_paths import configuration, SECRETS


def selected_provider():
    return configuration()['image_provider']


def client(provider=None):
    provider = provider or selected_provider()
    modules = {'company': 'studio.company_image_api', 'grsai': 'studio.image_api'}
    if provider not in modules:
        raise ValueError('Unknown image provider')
    return importlib.import_module(modules[provider])


def preflight(provider=None, model='gpt-image-2'):
    provider = provider or selected_provider()
    api = client(provider)
    api.model_preflight(model)
    return api.read_key(SECRETS) if provider == 'company' else api.read_key(SECRETS, model)


def status():
    provider = selected_provider()
    try:
        preflight(provider)
        return {'provider': provider, 'ready': True}
    except Exception as error:
        # Preflight errors are local configuration diagnostics, never API replies.
        return {'provider': provider, 'ready': False, 'message': str(error)}
