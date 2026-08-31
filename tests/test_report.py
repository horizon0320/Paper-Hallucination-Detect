from tools_generate_report import has_author_mismatch


def test_report_classifies_strong_title_with_conflicting_authors_as_suspected():
    item = {
        "overall_verdict": "METADATA_MISMATCH",
        "reference": {
            "authors": ["Minseok Kwon", "Sungmin Yu", "Hyungyu Kim"],
        },
        "checks": [{
            "evidence": {
                "title_score": 0.885,
                "best_candidate_authors": ["Myung-Joon Kwon", "In-Jae Yu", "Seung-Hun Nam"],
                "mismatches": ["authors: author list differs after abbreviation normalization"],
            }
        }],
    }
    assert has_author_mismatch(item)


def test_report_does_not_promote_low_ranked_not_found_candidate_to_suspected():
    item = {
        "overall_verdict": "NOT_FOUND",
        "reference": {"authors": ["Invented Author", "Another Person", "Third Person"]},
        "checks": [{
            "evidence": {
                "title_score": 0.42,
                "best_candidate_authors": ["Someone Else"],
            }
        }],
    }
    assert not has_author_mismatch(item)
