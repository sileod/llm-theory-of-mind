import pandas as pd

from mindgames.check import check
from mindgames.evaluate import parse
from mindgames.release import fix

MUDDY = "VARS 1,2 LAW Top OBS Agenta:2 Agentb:1 VALID? [ ! (1|2) ] "


def test_muddy_children():
    # After "someone is muddy", a muddy child seeing a clean one knows; nobody knowing leaves only "both muddy".
    assert check(MUDDY + "(Agenta knows whether 1)") == (0, 3)
    assert check(MUDDY + "[ ! ~(Agenta knows whether 1) & ~(Agentb knows whether 2) ] (Agenta knows that 1)") == (1, 1)


def test_contradictory_announcement_leaves_no_world():
    assert check(MUDDY + "[ ! ~(1|2) ] 0".replace("VARS 1,2", "VARS 0,1,2")) == (1, 0)


def test_nested_knowledge():
    # Everyone sees everyone (mirror): after any announcement, a knows that b knows whether 1.
    assert check("VARS 1,2 LAW Top OBS Agenta:1,2 Agentb:1,2 VALID? [ ! (1|2) ] Agenta knows that (Agentb knows whether 1)")[0] == 1


def test_explicit_observations_follow_the_text():
    # Agent b observes nothing; 1.0 shifted c's observation onto b.
    row = {"index": 0, "setup": "explicit", "n_agents": 3, "names": ["Ruth", "Dena", "Terrence"],
           "premise": "Terrence's card is shown to Ruth. Ruth's card is shown to Terrence.",
           "smcdel_problem": "VARS 1,2,3 LAW Top OBS Agenta:3 Agentb:1 VALID? [ ! (1|2|3) ] (Agentc knows whether 1)",
           "pbcheck": "VARS 0,1,2,3 LAW Top OBS Agenta:3 Agentb:1 VALID? [ ! (1|2|3) ] 0",
           "label": "not_entailment", "deberta_pred": 0, "deberta_confidence": 0.9}
    fixed, log = fix(pd.DataFrame([row]))
    assert "OBS Agenta:3 Agentc:1 VALID?" in fixed.smcdel_problem[0]
    assert fixed.label[0] == "entailment" and log == [{"index": 0, "change": "label not_entailment -> entailment"}]


def test_parse():
    assert parse("Yes") is True and parse("**Answer: No**") is False
    assert parse("reasoning...\n\nAnswer: Yes") is True and parse(r"\boxed{\text{No}}") is False
    assert parse(None) is None and parse("maybe") is None


def test_english_matches_labels():
    from mindgames.diamond import candidate
    from mindgames.english import parse_item

    items = [c for c in map(candidate, range(200)) if c]
    assert items
    for item in items:
        rebuilt = check(parse_item(item["premise"], item["hypothesis"], item["names"]))[0]
        assert ("entailment" if rebuilt else "not_entailment") == item["label"], item["premise"]
