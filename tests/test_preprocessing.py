from src.preprocessing import clean_text


def test_clean_text():

    text = "Hello!!! Visit https://test.com"

    cleaned = clean_text(text)

    assert "https" not in cleaned

    assert "!" not in cleaned