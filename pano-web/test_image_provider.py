import unittest
from unittest.mock import Mock, patch
import image_provider as providers
from project_paths import ROOT, MODELS, SHARP, DEPTHMASTER, LINGBOT, REPAIR, PROVIDERS


class ProviderSelectionTests(unittest.TestCase):
    def test_both_providers_are_local_and_distinct(self):
        company=providers.client('company');grsai=providers.client('grsai')
        self.assertIsNot(company,grsai)
        self.assertEqual(company.ENDPOINT_HOST,'oneapi.qunhequnhe.com')
        self.assertEqual(grsai.ENDPOINTS['gpt-image-2'][0],'grsaiapi.com')
        from pathlib import Path
        self.assertTrue(Path(company.__file__).resolve().is_relative_to(ROOT))
        self.assertTrue(Path(grsai.__file__).resolve().is_relative_to(ROOT))

    def test_explicit_selection_does_not_fall_back_to_another_provider(self):
        fake=Mock();fake.model_preflight.side_effect=RuntimeError('configured provider unavailable')
        with patch.object(providers,'client',return_value=fake) as select:
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                providers.preflight('company')
            select.assert_called_once_with('company')
            fake.read_key.assert_not_called()

    def test_grs_and_company_use_their_respective_key_contracts(self):
        for provider,expected in [('company',(providers.SECRETS,)),('grsai',(providers.SECRETS,'gpt-image-2'))]:
            fake=Mock();fake.read_key.return_value='test-only'
            with patch.object(providers,'client',return_value=fake):
                self.assertEqual(providers.preflight(provider),'test-only')
                fake.read_key.assert_called_once_with(*expected)

    def test_runtime_source_paths_are_inside_checkout(self):
        for path in (MODELS,SHARP,DEPTHMASTER,LINGBOT,REPAIR,PROVIDERS):
            self.assertTrue(path.resolve().is_relative_to(ROOT))
            self.assertTrue(path.is_dir())


if __name__=='__main__':
    unittest.main()
