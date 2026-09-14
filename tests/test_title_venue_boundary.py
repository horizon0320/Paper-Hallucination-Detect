"""Carrier-name tokens must not stay glued to the end of a parsed title."""

from paperaudit.parse.plaintext import _extract_author_year


def test_isprs_prefix_belongs_to_the_venue():
    chunk = (
        "Cheng, G.; Han, J.; Zhou, P.; and Guo, L. 2014. Multiclass geospatial "
        "object detection and geographic image classification based on "
        "collection of part detectors. ISPRS Journal of Photogrammetry and "
        "Remote Sensing."
    )
    title, _authors, venue = _extract_author_year(chunk)
    assert title == (
        "Multiclass geospatial object detection and geographic image "
        "classification based on collection of part detectors"
    )
    assert venue == "ISPRS Journal of Photogrammetry and Remote Sensing"


def test_international_prefix_belongs_to_the_venue():
    chunk = (
        "Li, X.; Wen, C.; Hu, Y.; and Zhou, N. 2023. RS-CLIP: Zero shot remote "
        "sensing scene classification via contrastive vision-language "
        "supervision. International Journal of Applied Earth Observation and "
        "Geoinformation."
    )
    title, _authors, venue = _extract_author_year(chunk)
    assert title == (
        "RS-CLIP: Zero shot remote sensing scene classification via "
        "contrastive vision-language supervision"
    )
    assert venue == (
        "International Journal of Applied Earth Observation and Geoinformation"
    )


def test_openreview_in_front_of_venue_stays_with_the_carrier():
    chunk = (
        "Wei, J. and Wang, X. 2024. Emergent abilities of large language "
        "models. In Transactions on Machine Learning Research "
        "(openreview.net/forum?id=oI0983lGoG)."
    )
    title, _authors, venue = _extract_author_year(chunk)
    assert title == "Emergent abilities of large language models"
    assert venue is not None
    assert venue.startswith("In Transactions on Machine Learning Research")


def test_multi_word_title_is_left_untouched():
    chunk = (
        "Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using "
        "t-SNE. JMLR, 9(11)."
    )
    title, _authors, _venue = _extract_author_year(chunk)
    assert title == "Visualizing data using t-SNE"


def test_acronym_inside_a_title_survives_a_sentence_period():
    chunk = (
        "Smith, A.; and Jones, B. 2021. Comparing CNN and RNN models for "
        "text. The design is simple and robust. Journal of Artificial "
        "Intelligence Research."
    )
    title, _authors, venue = _extract_author_year(chunk)
    assert "CNN and RNN" in title
    assert venue is not None
    assert "Journal of Artificial Intelligence Research" in venue
