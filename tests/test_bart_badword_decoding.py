import torch
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, set_seed
import random

set_seed(0)

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


def test_banned_words_decoding():

    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    # load bart tokenizer
    # with open(
    #         '/home/alvinwatner/alvin_research/topics/rc/qg/src/modified_bad_word_decoding/transformers/tests/test_samples/bart_tokenizer.obj',
    #         'rb') as file_path:
    #     tokenizer = pickle.load(file_path)

    config = BartConfig(
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        decoder_layers=1,
        decoder_ffn_dim=64,
        decoder_attention_heads=4,
        d_model=128
    )

    model = BartForConditionalGeneration(config)
    # model = BartForConditionalGeneration.from_pretrained(model_name)

    input_context = "My cute dog"
    inputs = tokenizer(input_context, return_tensors="pt")
    inputs['num_beams'] = 1


    banned_words = ["visitor complain", "lambda"]
    pack_of_banned_words_ids = banned_words_gpt2_tokenizer(banned_words, tokenizer)
    banned_words_ids = joint_pack_of_banned_words(pack_of_banned_words_ids)
    banned_words = {'ids': banned_words_ids, 'epsilon': 1.0}

    outputs = model.generate(**inputs, max_length=20, banned_words=banned_words, do_sample=False)

    print(f"Raw Generated = {outputs[0]}")
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    # for each_banned_word in pack_of_banned_words:
    #     for type_of_ids, ids in each_banned_word.items():
    #         print(f"type_of_ids = {type_of_ids} | ids = {ids}")
    #         decoded = [tokenizer.convert_ids_to_tokens(id) for id in ids]
    #         print(f"decoded_type_of_ids = {decoded}")
    #         print()
    #         empty_list.append(ids)
    #     print()
    #
    # print()
    # print(banned_words)


test_banned_words_decoding()