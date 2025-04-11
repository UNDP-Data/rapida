from cbsurge.session import Session
import os


def test_session():
    with Session() as s:
        s.set_account_name("test_account")
        s.set_file_share_name("test_share")

        assert s.get_blob_service_account_url() == "https://test_account.blob.core.windows.net"
        assert s.get_file_share_account_url() == "https://test_account.file.core.windows.net"

        assert s.get_blob_service_account_url(account_name="aaa") == "https://aaa.blob.core.windows.net"
        assert s.get_file_share_account_url(account_name="aaa") == "https://aaa.file.core.windows.net"
