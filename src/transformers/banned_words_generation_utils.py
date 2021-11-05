class Timesteps():
    """
    This class designed for ``banned words decoding`` to manage the timestep if
    any of the specified banned_words get produced by the model.
    """
    def __init__(self):
        self.timesteps_info = {'timesteps': [], 'sorted_scores_indices': [], 'token_idx': []}
        self.revert_input_ids = None

    def revert_timestep(self):
        """
        This function used to revert the timestep (input_ids) and shift the next_tokens
        on the reverted timestep. The shifted value controlled by ``token_idx``.
        """
        timestep = len(self.revert_input_ids[0])
        idx = self.timesteps_info['timesteps'].index(timestep)

        token_idx = self.timesteps_info['token_idx'][idx]
        sorted_scores_indices = self.timesteps_info['sorted_scores_indices'][idx]
        next_tokens = sorted_scores_indices[0, token_idx]
        input_ids = self.revert_input_ids
        self.revert_input_ids = None

        return input_ids, next_tokens

    def init_timestep(self, timestep, sorted_next_token_indices):
        """
        This function used to initialize timestep
        """
        self.timesteps_info['timesteps'].append(timestep)
        init_probs_idx = 1
        self.timesteps_info['token_idx'].append(init_probs_idx)
        self.timesteps_info['sorted_scores_indices'].append(sorted_next_token_indices)

    def update(self, input_ids, sorted_next_token_indices):
        """
        This function only get executed when the
        ``next_tokens`` match to the ``banned_words['ids'][0]``.
        """

        timestep = len(input_ids[0])
        if timestep not in self.timesteps_info['timesteps']:
            self.init_timestep(timestep, sorted_next_token_indices)
        else:
            idx = self.timesteps_info['timesteps'].index(timestep)
            self.timesteps_info['token_idx'][idx] += 1

        self.revert_input_ids = input_ids


def joint_pack_of_banned_words(pack_of_banned_words):
    concatenation_of_the_entire_banned_words = []
    for each_banned_word in pack_of_banned_words:
        for type_of_ids, ids in each_banned_word.items():
            concatenation_of_the_entire_banned_words.append(ids)

    return concatenation_of_the_entire_banned_words


def uppercase_first_letter(lowercase_words):
  sentence = ""

  for idx, word in enumerate(lowercase_words):
    if idx == 0:
      sentence += word.title()
    else:
      word = " " + word
      sentence += word

  return sentence


def banned_words_gpt2_tokenizer(banned_words, tokenizer):
    pack_of_banned_words = []
    for words in banned_words:

        lowercase_sentence = words.lower()
        splitted_lowercase_sentence = lowercase_sentence.split(" ")
        title_sentence = uppercase_first_letter(splitted_lowercase_sentence)

        """
        front_sentence_lower = The banned words appear at the front 
        (without prefix_space on the first word and with prefix_space on the rest)
        for e.g., ["my name is martin, bla bla bla."]    
        """
        front_sentence_lower_ids = tokenizer(lowercase_sentence).input_ids[1:-1]
        front_sentence_lower_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in front_sentence_lower_ids]
        # print(f"lowercase_sentence = {lowercase_sentence}")
        # print(f"front_sentence_lower_ids = {front_sentence_lower_ids}")
        # print(f"front_sentence_lower_tokens = {front_sentence_lower_tokens}")

        """
        front_sentence = The banned words appear at the front, with uppercase on the letter of the first word
        (without prefix_space on the first word and with prefix_space on the rest). 
        for e.g., ["My name is martin, bla bla bla."]    
        """
        front_sentence_ids = tokenizer(title_sentence).input_ids[1:-1]
        front_sentence_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in front_sentence_ids]
        # print(f"front_sentence = {title_sentence}")
        # print(f"front_sentence_ids = {front_sentence_ids}")
        # print(f"front_sentence_tokens = {front_sentence_tokens}")

        """
        middle_sentence = the banned words appear in the middle of the sentence with uppercase on the first word,
        and with prefix_space.         
        for e.g., ["bla bla bla bla. My name is martin"] 
        """
        middle_sentence = " " + title_sentence
        middle_sentence_ids = tokenizer(middle_sentence).input_ids[1:-1]
        middle_sentence_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in middle_sentence_ids]
        # print(f"middle_sentence = {middle_sentence}")
        # print(f"middle_sentence_ids = {middle_sentence_ids}")
        # print(f"middle_sentence_tokens = {middle_sentence_tokens}")

        """
        middle_sentence_lower = the banned words appear in the middle of the sentence and with prefix_space 
        for e.g., ["bla bla bla bla, my name is martin"] 
        """
        middle_sentence_lower = " " + lowercase_sentence
        middle_sentence_lower_ids = tokenizer(middle_sentence_lower).input_ids[1:-1]
        middle_sentence_lower_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in middle_sentence_lower_ids]
        # print(f"middle_sentence = {middle_sentence_lower}")
        # print(f"middle_sentence_lower_ids = {middle_sentence_lower_ids}")
        # print(f"middle_sentence_lower_tokens = {middle_sentence_lower_tokens}")
        # print()
        # print()

        pack_of_banned_words.append({'front_sentence_lower_ids' : front_sentence_lower_ids,
                                     'front_sentence_ids' : front_sentence_ids,
                                     'middle_sentence_lower_ids' : middle_sentence_lower_ids,
                                     'middle_sentence_ids' : middle_sentence_ids})

    return pack_of_banned_words