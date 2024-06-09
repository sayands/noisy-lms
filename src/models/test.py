import torch
from language import BartLargeCNNSummarization, LanguageModel
from reward import DeBERTaV3LargeV2, RewardModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summarizer: LanguageModel = BartLargeCNNSummarization(device=device)
    rm: RewardModel = DeBERTaV3LargeV2(device=device)

    story = """
    Once upon a time, in a faraway land,
    there lived a brave knight named Sir John.
    He was known for his courage and honor,
    and his unwavering dedication to justice.

    One day, a distress call reached Sir John.
    The princess of the kingdom had been kidnapped
    by an evil sorcerer who lived in a dark castle.
    Without hesitation, Sir John set off on a quest
    to rescue the princess and bring her back safely.

    As he journeyed through treacherous forests
    and crossed perilous rivers,
    Sir John encountered various challenges.
    He fought fierce monsters, solved intricate puzzles,
    and outsmarted cunning traps.

    After days of relentless pursuit,
    Sir John finally reached the castle.
    He faced the sorcerer in a fierce battle,
    using his sword and shield with great skill.
    With each strike, he grew closer to victory.

    In the end, Sir John emerged triumphant.
    He rescued the princess and returned her
    to the grateful kingdom.
    The people hailed him as a hero,
    and his name became legend.

    And so, Sir John's story became a tale
    that would be told for generations to come.
    His bravery and selflessness inspired many,
    and his legacy lived on in the hearts of all.

    The end.
    """

    good_summary = summarizer.generate(story)
    bad_summary = "Lightning struck the castle, and the princess was never seen again."

    good_score = rm.get_score(story, good_summary)
    bad_score = rm.get_score(story, bad_summary)

    print(f"{good_summary} -> {good_score}")
    print(f"{bad_summary} -> {bad_score}")
    assert good_score > bad_score
