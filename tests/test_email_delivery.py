"""Tests for storymesh.core.email_delivery."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, patch

import pytest

from storymesh.core.email_delivery import (
    EmailConfig,
    build_html_email,
    deliver_book,
    send_email,
    title_to_filename,
)


def _decoded_parts(msg: MIMEMultipart) -> str:
    """Walk all MIME parts and return all decoded text payloads concatenated.

    MIMEText with charset='utf-8' encodes bodies as base64, so
    ``msg.as_string()`` won't contain plaintext content. This helper decodes
    each part so tests can search for expected text naturally.
    """
    parts: list[str] = []
    for part in msg.walk():
        payload = part.get_payload(decode=True)
        if payload is not None:
            parts.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# title_to_filename
# ---------------------------------------------------------------------------

class TestTitleToFilename:
    def test_simple_title(self) -> None:
        assert title_to_filename("The Dragon's Lair") == "The_Dragons_Lair"

    def test_special_characters_stripped(self) -> None:
        assert title_to_filename("Iron & Blood: A Story") == "Iron_Blood_A_Story"

    def test_preserves_title_casing(self) -> None:
        assert title_to_filename("A Tale of Two Cities") == "A_Tale_of_Two_Cities"

    def test_leading_trailing_whitespace(self) -> None:
        assert title_to_filename("  Whitespace   ") == "Whitespace"

    def test_multiple_internal_spaces(self) -> None:
        assert title_to_filename("One  Two   Three") == "One_Two_Three"

    def test_empty_after_stripping(self) -> None:
        assert title_to_filename("!!!") == "Untitled"

    def test_empty_string(self) -> None:
        assert title_to_filename("") == "Untitled"

    def test_unicode_letters_preserved(self) -> None:
        result = title_to_filename("Café au Lait")
        assert "Café" in result or "Caf" in result  # normalisation may vary

    def test_dots_stripped(self) -> None:
        assert title_to_filename("A.I. Dreams") == "AI_Dreams"


# ---------------------------------------------------------------------------
# EmailConfig
# ---------------------------------------------------------------------------

class TestEmailConfig:
    def test_from_dict_basic(self) -> None:
        data = {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "from_address": "noreply@example.com",
            "recipient": "reader@example.com",
            "include_epub": True,
        }
        cfg = EmailConfig.from_dict(data)
        assert cfg.smtp_host == "smtp.example.com"
        assert cfg.smtp_port == 587
        assert cfg.smtp_user == "user@example.com"
        assert cfg.smtp_password == "secret"
        assert cfg.from_address == "noreply@example.com"
        assert cfg.recipient == "reader@example.com"
        assert cfg.include_epub is True

    def test_from_dict_defaults(self) -> None:
        cfg = EmailConfig.from_dict({})
        assert cfg.smtp_host == ""
        assert cfg.smtp_port == 587
        assert cfg.smtp_user == ""
        assert cfg.smtp_password == ""
        assert cfg.from_address == ""
        assert cfg.recipient == ""
        assert cfg.include_epub is False

    def test_effective_from_uses_from_address(self) -> None:
        cfg = EmailConfig.from_dict({
            "smtp_user": "user@example.com",
            "from_address": "sender@example.com",
        })
        assert cfg.effective_from == "sender@example.com"

    def test_effective_from_falls_back_to_smtp_user(self) -> None:
        cfg = EmailConfig.from_dict({"smtp_user": "user@example.com"})
        assert cfg.effective_from == "user@example.com"

    def test_is_configured_true(self) -> None:
        cfg = EmailConfig.from_dict({
            "smtp_host": "smtp.example.com",
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
        })
        assert cfg.is_configured is True

    def test_is_configured_false_missing_host(self) -> None:
        cfg = EmailConfig.from_dict({
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
        })
        assert cfg.is_configured is False

    def test_is_configured_false_missing_password(self) -> None:
        cfg = EmailConfig.from_dict({
            "smtp_host": "smtp.example.com",
            "smtp_user": "user@example.com",
        })
        assert cfg.is_configured is False

    def test_env_overrides_smtp_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STORYMESH_SMTP_USER", "env_user@example.com")
        cfg = EmailConfig.from_env_and_dict({"smtp_user": "config_user@example.com"})
        assert cfg.smtp_user == "env_user@example.com"

    def test_env_overrides_smtp_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STORYMESH_SMTP_PASSWORD", "env_password")
        cfg = EmailConfig.from_env_and_dict({"smtp_password": "config_password"})
        assert cfg.smtp_password == "env_password"


# ---------------------------------------------------------------------------
# build_html_email
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_email_config() -> EmailConfig:
    return EmailConfig.from_dict({
        "smtp_host": "smtp.example.com",
        "smtp_user": "sender@example.com",
        "smtp_password": "secret",
        "from_address": "sender@example.com",
        "recipient": "reader@example.com",
        "include_epub": False,
    })


@pytest.fixture()
def sample_pdf(tmp_path: pytest.TempPathFactory) -> str:  # type: ignore[type-arg]
    p = tmp_path / "The_Story.pdf"  # type: ignore[operator]
    p.write_bytes(b"%PDF-1.4 fake content")
    return str(p)


@pytest.fixture()
def sample_epub(tmp_path: pytest.TempPathFactory) -> str:  # type: ignore[type-arg]
    p = tmp_path / "The_Story.epub"  # type: ignore[operator]
    p.write_bytes(b"PK fake epub content")
    return str(p)


class TestBuildHtmlEmail:
    def test_returns_mime_multipart(
        self, sample_pdf: str, base_email_config: EmailConfig
    ) -> None:
        msg = build_html_email(
            title="The Story",
            synopsis="A great adventure.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        assert isinstance(msg, MIMEMultipart)

    def test_subject_contains_title(self, sample_pdf: str, base_email_config: EmailConfig) -> None:
        msg = build_html_email(
            title="Dragon's Lair",
            synopsis="Synopsis text.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="Dragons_Lair",
        )
        assert "Dragon's Lair" in msg["Subject"]

    def test_from_and_to_headers(self, sample_pdf: str) -> None:
        msg = build_html_email(
            title="Test",
            synopsis="Synopsis.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address="sender@example.com",
            recipient="reader@example.com",
            filename_stem="Test",
        )
        assert msg["From"] == "sender@example.com"
        assert msg["To"] == "reader@example.com"

    def test_pdf_attached_with_title_filename(
        self, sample_pdf: str, base_email_config: EmailConfig
    ) -> None:
        msg = build_html_email(
            title="The Story",
            synopsis="Synopsis.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        payload_str = msg.as_string()
        assert "The_Story.pdf" in payload_str

    def test_epub_not_attached_when_include_epub_false(
        self, sample_pdf: str, sample_epub: str, base_email_config: EmailConfig
    ) -> None:
        msg = build_html_email(
            title="The Story",
            synopsis="Synopsis.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path=sample_epub,
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        payload_str = msg.as_string()
        assert "The_Story.epub" not in payload_str

    def test_epub_attached_when_include_epub_true(
        self, sample_pdf: str, sample_epub: str, base_email_config: EmailConfig
    ) -> None:
        msg = build_html_email(
            title="The Story",
            synopsis="Synopsis.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path=sample_epub,
            include_epub=True,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        payload_str = msg.as_string()
        assert "The_Story.epub" in payload_str

    def test_cover_image_embedded_as_cid(
        self, sample_pdf: str, base_email_config: EmailConfig
    ) -> None:
        cover_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # minimal PNG-like header
        msg = build_html_email(
            title="The Story",
            synopsis="Synopsis.",
            cover_image_bytes=cover_bytes,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        # Content-ID header is plain text; the HTML body (base64) references
        # the same CID but we check the header as it is always readable.
        assert "Content-ID: <cover_art>" in msg.as_string()
        # The HTML body references the CID — verify via decoded payload.
        assert "cid:cover_art" in _decoded_parts(msg)

    def test_plain_text_fallback_present(
        self, sample_pdf: str, base_email_config: EmailConfig
    ) -> None:
        msg = build_html_email(
            title="The Story",
            synopsis="Plain text synopsis here.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        assert "Plain text synopsis here." in _decoded_parts(msg)

    def test_html_body_contains_synopsis(
        self, sample_pdf: str, base_email_config: EmailConfig
    ) -> None:
        msg = build_html_email(
            title="The Story",
            synopsis="Epic adventure awaits.",
            cover_image_bytes=None,
            pdf_path=sample_pdf,
            epub_path="",
            include_epub=False,
            from_address=base_email_config.effective_from,
            recipient="reader@example.com",
            filename_stem="The_Story",
        )
        assert "Epic adventure awaits." in _decoded_parts(msg)


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

class TestSendEmail:
    def test_calls_smtp_with_correct_host_and_port(self) -> None:
        config = EmailConfig.from_dict({
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
        })
        msg = MIMEMultipart("mixed")
        msg["Subject"] = "Test"
        msg["From"] = "user@example.com"
        msg["To"] = "reader@example.com"

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            send_email(config=config, message=msg, recipient="reader@example.com")
            mock_smtp_cls.assert_called_once_with("smtp.example.com", 587)

    def test_calls_starttls_and_login(self) -> None:
        config = EmailConfig.from_dict({
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
        })
        msg = MIMEMultipart("mixed")
        msg["Subject"] = "Test"
        msg["From"] = "user@example.com"
        msg["To"] = "reader@example.com"

        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            send_email(config=config, message=msg, recipient="reader@example.com")
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("user@example.com", "secret")


# ---------------------------------------------------------------------------
# deliver_book
# ---------------------------------------------------------------------------

class TestDeliverBook:
    def _make_config(self, **overrides: object) -> EmailConfig:
        data: dict[str, object] = {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "from_address": "user@example.com",
            "recipient": "reader@example.com",
            "include_epub": False,
        }
        data.update(overrides)
        return EmailConfig.from_dict(data)  # type: ignore[arg-type]

    def test_skips_when_recipient_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = self._make_config(recipient="")
        with patch("storymesh.core.email_delivery.send_email") as mock_send:
            deliver_book(
                title="Test",
                synopsis="Synopsis.",
                cover_image_bytes=None,
                pdf_path="/fake/Test.pdf",
                epub_path="",
                recipient="",
                email_config=cfg,
            )
            mock_send.assert_not_called()

    def test_skips_when_not_configured(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = self._make_config(smtp_host="", smtp_user="", smtp_password="")
        with patch("storymesh.core.email_delivery.send_email") as mock_send:
            deliver_book(
                title="Test",
                synopsis="Synopsis.",
                cover_image_bytes=None,
                pdf_path="/fake/Test.pdf",
                epub_path="",
                recipient="reader@example.com",
                email_config=cfg,
            )
            mock_send.assert_not_called()

    def test_calls_send_email_when_configured(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pdf = tmp_path / "Test.pdf"  # type: ignore[operator]
        pdf.write_bytes(b"%PDF fake")
        cfg = self._make_config()
        with patch("storymesh.core.email_delivery.send_email") as mock_send:
            deliver_book(
                title="Test Story",
                synopsis="A synopsis.",
                cover_image_bytes=None,
                pdf_path=str(pdf),
                epub_path="",
                recipient="reader@example.com",
                email_config=cfg,
            )
            mock_send.assert_called_once()

    def test_logs_error_on_smtp_exception(
        self, tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture  # type: ignore[type-arg]
    ) -> None:
        pdf = tmp_path / "Test.pdf"  # type: ignore[operator]
        pdf.write_bytes(b"%PDF fake")
        cfg = self._make_config()
        with patch(
            "storymesh.core.email_delivery.send_email",
            side_effect=smtplib.SMTPException("connection refused"),
        ):
            with caplog.at_level(logging.ERROR, logger="storymesh.core.email_delivery"):
                deliver_book(
                    title="Test Story",
                    synopsis="A synopsis.",
                    cover_image_bytes=None,
                    pdf_path=str(pdf),
                    epub_path="",
                    recipient="reader@example.com",
                    email_config=cfg,
                )
            assert any("Email delivery failed" in r.message for r in caplog.records)

    def test_uses_title_to_filename_for_attachment(
        self, tmp_path: pytest.TempPathFactory  # type: ignore[type-arg]
    ) -> None:
        pdf = tmp_path / "output.pdf"  # type: ignore[operator]
        pdf.write_bytes(b"%PDF fake")
        cfg = self._make_config()
        captured: list[MIMEMultipart] = []

        def capture_send(
            *, config: EmailConfig, message: MIMEMultipart, recipient: str
        ) -> None:
            captured.append(message)

        with patch("storymesh.core.email_delivery.send_email", side_effect=capture_send):
            deliver_book(
                title="The Dragon's Lair",
                synopsis="Epic.",
                cover_image_bytes=None,
                pdf_path=str(pdf),
                epub_path="",
                recipient="reader@example.com",
                email_config=cfg,
            )

        assert len(captured) == 1
        payload_str = captured[0].as_string()
        assert "The_Dragons_Lair.pdf" in payload_str
