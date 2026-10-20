"""Default provider TLS must retain verification without trusting the compromised emSign root."""

import hashlib
import ssl

import certifi

from ml4t.data.providers import SyntheticProvider

ISRG_ROOT_X1_SHA256 = "96bcec06264976f37460779acf28c5a7cfe8a3c0aae11a8ffcee05c0bddf08c6"
COMPROMISED_EMSIGN_G3_SHA256 = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"


def test_default_provider_tls_client_does_not_trust_compromised_emsign_g3(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    with SyntheticProvider() as provider:
        context = provider.session._transport._pool._ssl_context
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        roots = set()
        for block in certifi.contents().split("-----BEGIN CERTIFICATE-----")[1:]:
            pem = "-----BEGIN CERTIFICATE-----" + block.split("-----END CERTIFICATE-----")[0]
            der = ssl.PEM_cert_to_DER_cert(pem + "-----END CERTIFICATE-----")
            roots.add(hashlib.sha256(der).hexdigest())
        assert ISRG_ROOT_X1_SHA256 in roots
        assert COMPROMISED_EMSIGN_G3_SHA256 not in roots
