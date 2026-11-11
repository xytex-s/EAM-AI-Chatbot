"""
safety.py - wellbeing safety check (Ahmed)

/ask calls this FIRST, before search and the language model. If a question
suggests someone may be at risk, /ask skips the AI completely and returns the
fixed message and links below, which were written by people.

    check(question: str) -> dict | None
        None -> a normal question, so /ask carries on
        {"message": str, "sources": [{"title": str, "url": str}, ...]}
             -> /ask returns this with "safety": true

How it works:
- Each keyword group maps to one SHU wellbeing page. "Urgent help" is always
  the first link. If several groups match, every matching page is included.
- Matching ignores capitals and punctuation, so "Self-harm", "self harm" and
  "SELF HARM" all match, and "don't" matches "dont". Phrases only match whole
  words, so "rape" doesn't match "grapes" and "therapist" is never a match.

Design choices (for the report):
- Keyword matching, not AI: predictable, instant, easy to test, and it can't
  invent a phone number.
- It errs on the side of triggering. A false alarm only shows support links,
  but a miss could mean a distressed student gets an AI answer.
- Bare words such as "alcohol", "drugs", "safe" or "crisis" are NOT used,
  because "Is alcohol allowed in halls?" or "cost of living crisis support"
  shouldn't show a crisis card. The cost: some real cases worded in other
  ways will be missed. Keyword matching can't understand meaning.

[CHECK] Every URL and the message text were copied from shu.ac.uk on
11 Nov 2026. Ahmed: re-check them on the live pages before the demo.
"""

import re

# Always the first link  [CHECK] shu.ac.uk/wellbeing/urgent-help
URGENT_HELP = {"title": "Urgent help", "url": "https://www.shu.ac.uk/wellbeing/urgent-help"}

# The fixed message. Wording copied from SHU's "Help in an emergency" page:
# [CHECK] https://www.shu.ac.uk/myhallam/help-in-an-emergency
MESSAGE = (
    "If you need urgent help, support is available. "
    "In an immediate or life-threatening emergency, call 999. "
    "On campus, call Security on 0114 225 2000 or 888 from an internal phone. "
    "The links below go to the university's own support pages."
)

TOPICS_URL = "https://www.shu.ac.uk/wellbeing/self-help/topics/"

# Keyword groups. "page" is None for groups that only need Urgent help.
# Titles and URLs are from the A-Z at shu.ac.uk/wellbeing/self-help/topics  [CHECK]
GROUPS = [
    {
        "name": "suicide_self_harm",
        "page": {"title": "Suicide", "url": TOPICS_URL + "suicide"},
        "phrases": [
            "suicide", "suicidal", "kill myself", "killing myself", "end my life",
            "ending my life", "end it all", "take my own life", "want to die",
            "wanna die", "dont want to live", "dont want to be alive",
            "dont want to be here anymore", "better off dead", "no reason to live",
            "not worth living", "self harm", "selfharm", "harm myself",
            "harming myself", "hurt myself", "hurting myself", "cut myself",
            "cutting myself", "overdose", "overdosed", "too many pills",
        ],
    },
    {
        "name": "eating_disorders",
        "page": {"title": "Eating disorders", "url": TOPICS_URL + "eating-disorders"},
        "phrases": [
            "eating disorder", "eating disorders", "anorexia", "anorexic",
            "bulimia", "bulimic", "binge eating", "purging", "make myself sick",
            "making myself sick", "made myself sick", "starving myself",
            "stopped eating", "arfid",
        ],
    },
    {
        "name": "domestic_abuse",
        "page": {"title": "Domestic and relationship abuse",
                 "url": TOPICS_URL + "domestic-and-relationship-abuse"},
        "phrases": [
            "domestic abuse", "domestic violence", "abusive relationship",
            "abusive partner", "abusive boyfriend", "abusive girlfriend",
            "abusive husband", "abusive wife", "hits me", "hit me again",
            "being abused", "abusing me", "coercive control",
            "controlling relationship", "scared of my partner",
            "afraid of my partner", "scared to go home",
        ],
    },
    {
        "name": "psychosis",
        "page": {"title": "Psychosis", "url": TOPICS_URL + "psychosis"},
        "phrases": [
            "psychosis", "psychotic", "hearing voices", "hear voices",
            "voices in my head", "voices telling me", "seeing things that arent there",
        ],
    },
    {
        "name": "drugs_alcohol",
        "page": {"title": "Drugs and alcohol", "url": TOPICS_URL + "drugs-and-alcohol"},
        # No bare "alcohol" or "drugs": "Is alcohol allowed in halls?" must not trigger.
        "phrases": [
            "overdose", "overdosed", "addicted", "addiction", "cant stop drinking",
            "drinking problem", "drug problem", "alcohol problem", "drink was spiked",
            "drink spiked", "been spiked", "got spiked", "was spiked", "spiked my drink",
        ],
    },
    {
        "name": "forced_marriage",
        "page": {"title": "Arranged or forced marriage",
                 "url": TOPICS_URL + "arranged-or-forced-marriage"},
        "phrases": [
            "forced marriage", "forced to marry", "forcing me to marry",
            "force me to marry", "made to marry", "make me marry", "making me marry",
            "married against my will",
        ],
    },
    {
        "name": "pregnancy",
        "page": {"title": "Pregnancy, miscarriage and abortion", "url": TOPICS_URL + "pregnancy"},
        # No bare "pregnant": "I'm pregnant, can I defer my course?" is an admin question.
        "phrases": [
            "miscarriage", "miscarried", "lost my baby", "lost the baby", "abortion",
            "terminate my pregnancy", "terminate the pregnancy",
            "termination of pregnancy", "unplanned pregnancy", "unwanted pregnancy",
        ],
    },
    {
        "name": "report_incident",
        # SHU's "Help in an emergency" page links here for sexual misconduct,
        # harassment, bullying, hate incidents and domestic abuse.  [CHECK]
        "page": {"title": "Report and Support", "url": "https://reportandsupport.shu.ac.uk/"},
        "phrases": [
            "sexual assault", "sexually assaulted", "assaulted", "raped", "rape",
            "sexual harassment", "sexually harassed", "harassing me", "being harassed",
            "hate crime", "hate incident", "being stalked", "stalking me",
            "being bullied", "bullying me", "drink was spiked", "spiked my drink",
        ],
    },
    {
        "name": "general_crisis",
        "page": None,    # Urgent help only
        "phrases": [
            "dont feel safe", "do not feel safe", "feel unsafe", "im not safe",
            "in danger", "cant cope", "cannot cope", "can not cope", "cant go on",
            "cant take it anymore", "cant take this anymore", "in crisis",
            "having a crisis", "mental health crisis", "having a breakdown",
            "mental breakdown", "against my will", "give up on life",
        ],
    },
]


def normalise(text: str) -> str:
    """Lower-case, drop apostrophes, turn punctuation into spaces.

    Pads with spaces so a phrase can be matched as whole words:
    "I can't cope!" -> " i cant cope "
    """
    text = text.lower().replace("’", "").replace("'", "")   # don't -> dont
    text = re.sub(r"[^a-z0-9]+", " ", text)                       # self-harm -> self harm
    return " " + " ".join(text.split()) + " "


# Normalise every phrase once, when the module loads
_COMPILED = [
    (group, [normalise(p) for p in group["phrases"]])
    for group in GROUPS
]


def matched_groups(question: str) -> list[str]:
    """Names of the keyword groups a question matches (used by the tests)."""
    if not isinstance(question, str):
        return []
    text = normalise(question)
    return [group["name"] for group, phrases in _COMPILED
            if any(p in text for p in phrases)]


def check(question: str) -> dict | None:
    """Return the fixed wellbeing response if the question triggers, else None."""
    names = matched_groups(question)
    if not names:
        return None

    sources = [URGENT_HELP]
    for group in GROUPS:
        page = group["page"]
        if group["name"] in names and page and page not in sources:
            sources.append(page)

    # Copies, so the caller can't accidentally change our constants
    return {"message": MESSAGE, "sources": [dict(s) for s in sources]}


if __name__ == "__main__":
    # Quick manual test: python safety.py "I don't feel safe right now"
    import json
    import sys
    question = " ".join(sys.argv[1:]) or "I don't feel safe right now"
    print(json.dumps(check(question), indent=2))
