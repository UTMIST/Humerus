class CardUtils:

    @classmethod
    def get_num_slots(cls, black_card) -> int:
        num_slots = 0

        for line in black_card:
            for e in line:
                if isinstance(e, dict):
                    num_slots += 1
        return num_slots

    @classmethod
    def combine(cls, black_card, white_cards) -> str:
        if CardUtils.get_num_slots(black_card) != len(white_cards):
            raise ValueError()

        sentence = ""
        white_card_index = 0

        for line in black_card:
            for e in line:
                if isinstance(e, str):
                    sentence += e
                    continue

                elif isinstance(e, dict):
                    white_card = white_cards[white_card_index]
                    if e.get('transform', '') == 'Capitalize':
                        white_card = white_card.capitalize()

                    sentence += white_card
                    white_card_index += 1

                    # TODO: actually, there is a case where two dictionaries
                    #  appear in a row (e.g. ['a ', {}, {}, '.']) In this case,
                    #  we need to add a space between them.

                else:
                    raise ValueError()

            sentence += " "

        return sentence[:-1]  # remove final blank space
