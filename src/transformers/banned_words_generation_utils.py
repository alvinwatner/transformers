import torch

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
        print(f"[REVERTED] to timestep : {timestep} with next_token : {next_tokens} shifted from "
              f"input_ids : {input_ids}")

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
            print(f"[INIT TIMESTEP] : {timestep} | sorted_next_token_indices : {sorted_next_token_indices}")
        else:
            idx = self.timesteps_info['timesteps'].index(timestep)
            self.timesteps_info['token_idx'][idx] += 1
            print(f"[NOT INIT TIMESTEP] : {self.timesteps_info['token_idx'][idx]}")

        self.revert_input_ids = input_ids


def joint_pack_of_banned_words(pack_of_banned_words):
    concatenation_of_the_entire_banned_words = []
    for each_banned_word in pack_of_banned_words:
        for type_of_ids, ids in each_banned_word.items():
            concatenation_of_the_entire_banned_words.append(ids)

    return concatenation_of_the_entire_banned_words

def lowercase_first_letter(words):
  sentence = ""

  for idx, word in enumerate(words):
    if idx == 0:
      sentence += word.lower()
    else:
      word = " " + word
      sentence += word

  return sentence


def uppercase_first_letter(lowercase_words):
  sentence = ""

  for idx, word in enumerate(lowercase_words):
    if idx == 0:
      sentence += word.title()
    else:
      word = " " + word
      sentence += word

  return sentence

def skip_bos_and_eos(tokenized_word_ids, tokenizer):
  eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
  bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
  if bos_token_id in tokenized_word_ids:
    tokenized_word_ids.remove(bos_token_id)
  if eos_token_id in tokenized_word_ids:
    tokenized_word_ids.remove(eos_token_id)
  return tokenized_word_ids


def banned_words_gpt2_tokenizer(banned_words, tokenizer):
    pack_of_banned_words = []
    for words in banned_words:

        splitted_raw_sentence = words.split(" ")
        print(f"len(splitted_raw_sentence) = {len(splitted_raw_sentence)}")

        lowercase_sentence = words.lower()
        splitted_lowercase_sentence = lowercase_sentence.split(" ")
        title_sentence = uppercase_first_letter(splitted_lowercase_sentence)

        """
        front_raw_sentence_upper = We uppercase the first word while keep the rest original
        note : raw sentence is useful when the subsequent token has abbreviation. 
        for e.g., ["My name is martin and I love KFC"]
        """
        front_raw_sentence_upper = uppercase_first_letter(splitted_raw_sentence)
        front_raw_sentence_upper_ids = tokenizer(front_raw_sentence_upper).input_ids
        front_raw_sentence_upper_ids = skip_bos_and_eos(front_raw_sentence_upper_ids, tokenizer)
        # front_raw_sentence_upper_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in front_raw_sentence_upper_ids]
        # print(f"front_raw_sentence_upper_ids = {front_raw_sentence_upper_ids}")
        # print(f"front_raw_sentence_upper_tokens = {front_raw_sentence_upper_tokens}")

        """
        front_raw_sentence = Keep everything as it is
        note : raw sentence is useful when the subsequent token has abbreviation. 
        for e.g., ["my name is martin and I love KFC"]                
        """
        front_raw_sentence_ids = tokenizer(words).input_ids
        front_raw_sentence_ids = skip_bos_and_eos(front_raw_sentence_ids, tokenizer)
        # front_raw_sentence_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in front_raw_sentence_ids]
        # print(f"front_raw_sentence_ids = {front_raw_sentence_ids}")
        # print(f"front_raw_sentence_tokens = {front_raw_sentence_tokens}")

        """
        front_sentence_lower = The banned words appear at the front 
        (without prefix_space on the first word and with prefix_space on the rest)
        for e.g., ["my name is martin, bla bla bla."]    
        """
        front_sentence_lower_ids = tokenizer(lowercase_sentence).input_ids
        front_sentence_lower_ids = skip_bos_and_eos(front_sentence_lower_ids, tokenizer)
        # front_sentence_lower_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in front_sentence_lower_ids]
        # print(f"lowercase_sentence = {lowercase_sentence}")
        # print(f"front_sentence_lower_ids = {front_sentence_lower_ids}")
        # print(f"front_sentence_lower_tokens = {front_sentence_lower_tokens}")

        """
        front_sentence = The banned words appear at the front, with uppercase on the letter of the first word
        (without prefix_space on the first word and with prefix_space on the rest). 
        for e.g., ["My name is martin, bla bla bla."]    
        """
        front_sentence_ids = tokenizer(title_sentence).input_ids
        front_sentence_ids = skip_bos_and_eos(front_sentence_ids, tokenizer)
        # front_sentence_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in front_sentence_ids]
        # print(f"front_sentence = {title_sentence}")
        # print(f"front_sentence_ids = {front_sentence_ids}")
        # print(f"front_sentence_tokens = {front_sentence_tokens}")


        """
        middle_raw_sentence_upper = The banned words that appear in the middle of the sentence
        note : raw sentence is useful when the subsequent token has abbreviation. 
        for e.g., ["bla bla bla bla. My name is martin and I love KFC"]
        """
        middle_raw_sentence_upper = " " + front_raw_sentence_upper
        middle_raw_sentence_upper_ids = tokenizer(middle_raw_sentence_upper).input_ids
        middle_raw_sentence_upper_ids = skip_bos_and_eos(middle_raw_sentence_upper_ids, tokenizer)
        # middle_raw_sentence_upper_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in middle_raw_sentence_upper_ids]
        # print(f"middle_raw_sentence_upper_ids = {middle_raw_sentence_upper_ids}")
        # print(f"middle_raw_sentence_upper_tokens = {middle_raw_sentence_upper_tokens}")

        """
        middle_raw_sentence = The banned words that appear in the middle of the sentence
        note : raw sentence is useful when the subsequent token has abbreviation. 
        for e.g., ["bla bla bla bla. my name is martin and I love KFC"] 
        """
        middle_raw_sentence = " " + words
        middle_raw_sentence_ids = tokenizer(middle_raw_sentence).input_ids
        middle_raw_sentence_ids = skip_bos_and_eos(middle_raw_sentence_ids, tokenizer)
        # middle_raw_sentence_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in middle_raw_sentence_ids]
        # print(f"middle_raw_sentence_ids = {middle_raw_sentence_ids}")
        # print(f"middle_raw_sentence_tokens = {middle_raw_sentence_tokens}")


        """
        middle_sentence = the banned words appear in the middle of the sentence with uppercase on the first word,
        and with prefix_space.         
        for e.g., ["bla bla bla bla. My name is martin"] 
        """
        middle_sentence = " " + title_sentence
        middle_sentence_ids = tokenizer(middle_sentence).input_ids
        middle_sentence_ids = skip_bos_and_eos(middle_sentence_ids, tokenizer)
        # middle_sentence_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in middle_sentence_ids]
        # print(f"middle_sentence = {middle_sentence}")
        # print(f"middle_sentence_ids = {middle_sentence_ids}")
        # print(f"middle_sentence_tokens = {middle_sentence_tokens}")

        """
        middle_sentence_lower = the banned words appear in the middle of the sentence and with prefix_space 
        for e.g., ["bla bla bla bla, my name is martin"] 
        """
        middle_sentence_lower = " " + lowercase_sentence
        middle_sentence_lower_ids = tokenizer(middle_sentence_lower).input_ids
        middle_sentence_lower_ids = skip_bos_and_eos(middle_sentence_lower_ids, tokenizer)
        # middle_sentence_lower_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in middle_sentence_lower_ids]
        # print(f"middle_sentence = {middle_sentence_lower}")
        # print(f"middle_sentence_lower_ids = {middle_sentence_lower_ids}")
        # print(f"middle_sentence_lower_tokens = {middle_sentence_lower_tokens}")
        # print()
        # print()

        pack_of_banned_words.append({'front_raw_sentence_ids': front_raw_sentence_ids,
                                     'front_raw_sentence_upper_ids': front_raw_sentence_upper_ids,
                                     'front_sentence_lower_ids' : front_sentence_lower_ids,
                                     'front_sentence_ids' : front_sentence_ids,
                                     'middle_raw_sentence_ids': middle_raw_sentence_ids,
                                     'middle_raw_sentence_upper_ids': middle_raw_sentence_upper_ids,
                                     'middle_sentence_lower_ids' : middle_sentence_lower_ids,
                                     'middle_sentence_ids' : middle_sentence_ids,
                                     })

    return pack_of_banned_words

class BannedWordsMechanism():
    def __init__(self, banned_words = None):
        if banned_words is not None:
            self.active = True
            self.epsilon = banned_words['epsilon']
            self.banned_words_ids = banned_words['ids']

            self.revert = False
            self.detected_banned_words_length_greater_than_1 = None
            self.timesteps = Timesteps()
        else:
            self.active = False

    def __call__(self):
        return self.active

    def process(self, next_tokens, sorted_next_token_indices, input_ids):
        random_uniform = torch.rand((1,))
        is_return_value = False

        if self.revert and self.epsilon > random_uniform:
            input_ids, next_tokens = self.timesteps.revert_timestep()
            self.revert = False
            is_return_value = True
        else:
            if self.detected_banned_words_length_greater_than_1 is not None:
                next_idx = self.detected_banned_words_length_greater_than_1['next_idx']

                if next_tokens != self.detected_banned_words_length_greater_than_1['ids'][next_idx]:
                    """
                    If the next_tokens is not equal to the subsequent token in the banned words,
                    we will set the detected banned_words to None. 
                    For e.g., banned_words = ['blue rabbits'], while the generated sequence
                    is "In the early monday, the blue sky ..."                    
                    """
                    print()
                    print(
                        f"[DETECTION CANCELED] :  The next_tokens {next_tokens} not equal to the banned_words[next_index] {detected_banned_words_length_greater_than_1['ids'][next_idx]}")
                    print(f"with the full banned_words = {self.detected_banned_words_length_greater_than_1['ids']}")
                    print()
                    self.detected_banned_words_length_greater_than_1 = None
                else:
                    if (self.detected_banned_words_length_greater_than_1['next_idx'] + 1) == \
                            len(self.detected_banned_words_length_greater_than_1['ids']):
                        print()
                        print(
                            f"[DETECTION FINISHED] :  The next_tokens {next_tokens} equal to the final tokens {self.detected_banned_words_length_greater_than_1['ids'][next_idx]}")
                        print(f"with the full banned_words = {self.detected_banned_words_length_greater_than_1['ids']}")
                        print()
                        self.revert = True
                        self.detected_banned_words_length_greater_than_1 = None
                    else:
                        print()
                        print(
                            f"[DETECTION CONTINUE] :  The next_tokens {next_tokens} equal to the banned_words[next_index] {self.detected_banned_words_length_greater_than_1['ids'][next_idx]}")
                        print(f"with the full banned_words = {self.detected_banned_words_length_greater_than_1['ids']}")
                        print()
                        self.detected_banned_words_length_greater_than_1['next_idx'] += 1

            else:
                for ids in self.banned_words_ids:
                    #  next_tokens.shape : (batch_size,)
                    for next_token in next_tokens:
                        print(f"[CHECKING] next_tokens = {next_token} with banned_word_ids[0] = {ids[0]}")
                        if next_token == ids[0]:
                            if len(ids) == 1:
                                print()
                                print("=" * 10)
                                print(f"[DETECTED] length 1 | next_tokens = {next_token} | ids = {ids}")
                                print("=" * 10)
                                print()
                                self.revert = True
                            else:
                                self.detected_banned_words_length_greater_than_1 = {'ids': ids,
                                                                                    'next_idx': 1}
                                print()
                                print("=" * 10)
                                print(f"[DETECTED] length > 1 | next_tokens = {next_token} | ids = {ids}")
                                print("=" * 10)
                                print()
                            self.timesteps.update(input_ids, sorted_next_token_indices)
        print()

        return is_return_value, input_ids, next_tokens