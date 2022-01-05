import torch
from typing import List, Dict, Any

class BannedWordsMechanism():
    def __init__(self, batch_size,
                 banned_words: Dict[str, Any] = None):

        if banned_words is None:
            self.active = False

        else:
            self.active = True

            self.epsilon = banned_words['epsilon']
            self.banned_words_ids = banned_words['ids']

            self.sequences_info = [{'batch_idx': idx,
                                    'input_ids_when_get_detected': None,
                                    'sorted_next_tokens_indices_when_get_detected': None,
                                    'next_tokens_when_get_detected': None,
                                    'next_banned_words_idx': None,
                                    'detected_banned_words_ids': [],
                                    'updated_timestep': [],
                                    'updated_next_tokens': {},
                                    'updated_input_ids': {},
                                    'updated_shifted_indices': {},
                                    'input_ids_when_detection_finished': None,
                                    'pause_tracking': False,
                                    'input_ids_when_paused': None
                                    }
                                   for idx in range(batch_size)]

            self.untracked_sequence = [sequence for sequence in self.sequences_info]
            self.tracked_sequence = []
            self.sequence_that_need_to_be_reverted = []
            self.is_currently_reverting_sequence = False
            self.start_revert = False

    def __call__(self):
        return self.active

    def prepare_to_revert(self, input_ids, sequence):
        """
        At this timestep, we will prepare to revert the given sequence and store the input_ids
        information for the reverting purpose. We also reset other informations that are no longer
        necessary as the sequence about to be reverted.
        """

        # This is to undo the `next_banned_words_idx` for every sequence
        # that has index smaller than the reverted sequence
        sequence_idx = self.tracked_sequence.index(sequence)
        if sequence_idx != 0:
            for i in range(0, sequence_idx):
                self.tracked_sequence[i]['next_banned_words_idx'] -= 1

        # step 1 : store the current input_ids to the sequence.
        sequence['input_ids_when_detection_finished'] = input_ids

        # step 2 : reset all the necessary things related to banned_words.
        sequence['detected_banned_words_ids'] = []
        sequence['next_banned_words_idx'] = None

        # step 3 : queue it into the `sequence_that_need_to_be_reverted` list.
        self.sequence_that_need_to_be_reverted.append(sequence)

        # step 4 : remove it from `tracked_sequence`.
        self.tracked_sequence.remove(sequence)

        # activated the `start_revert` to start revert.
        self.start_revert = True

        # For the other sequence in `tracked_sequence` that currently in the process
        # of tracking, we have to pause for a while until the reverting process finished.
        if self.tracked_sequence != 0:
            for i, seq in enumerate(self.tracked_sequence):
                batch_idx = seq['batch_idx']
                self.tracked_sequence[i]['pause_tracking'] = True
                self.tracked_sequence[i]['input_ids_when_paused'] = input_ids[batch_idx]

    def does_the_input_ids_and_next_tokens_ever_being_shifted(self, input_ids, next_tokens, current_timestep=None):
        """
        This function simply check if there is a sequence in the batch that has their next_tokens being
        shifted and has their input_ids being updated in previous timestep. If there is, we will
        replace it.
        (note : shifting is part of the reverting process)
        """
        is_input_ids_and_next_tokens_shifted_previously = False

        for sequence in self.sequences_info:
            if current_timestep in sequence['updated_timestep']:
                batch_idx = sequence['batch_idx']
                next_tokens[batch_idx] = sequence['updated_next_tokens'][f'{current_timestep}']
                input_ids[batch_idx] = sequence['updated_input_ids'][f'{current_timestep}']
                is_input_ids_and_next_tokens_shifted_previously = True

        return is_input_ids_and_next_tokens_shifted_previously, input_ids, next_tokens

    def untrack_sequence(self, sequence):
        """
        This function used by self.tracking to untrack the sequence that is no longer
        need to be tracked.
        """

        self.tracked_sequence.remove(sequence)
        sequence['input_ids_when_get_detected'] = None
        sequence['sorted_next_tokens_indices_when_get_detected'] = None
        sequence['detected_banned_words_ids'] = []
        sequence['next_banned_words_idx'] = None
        self.untracked_sequence.append(sequence)

    def init_revert(self):
        """
        This function get executed after the preparation for reverting sequence has
        complete. (see self.prepare_to_revert function)
        """

        sequence = self.sequence_that_need_to_be_reverted[0]

        batch_idx = sequence['batch_idx']
        reverted_input_ids = sequence['input_ids_when_get_detected']
        next_tokens = sequence['next_tokens_when_get_detected']
        sorted_next_token_indices = sequence['sorted_next_tokens_indices_when_get_detected']

        timestep = reverted_input_ids.shape[1]
        if timestep in self.sequences_info[batch_idx]['updated_timestep']:
            shifted_indices = self.sequences_info[batch_idx]['updated_shifted_indices'][f'{timestep}']
        else:
            shifted_indices = 1
            next_tokens[batch_idx] = sorted_next_token_indices[shifted_indices]

            self.sequences_info[batch_idx]['updated_timestep'].append(timestep)
            self.sequences_info[batch_idx]['updated_next_tokens'][f'{timestep}'] = next_tokens[batch_idx]
            self.sequences_info[batch_idx]['updated_input_ids'][f'{timestep}'] = reverted_input_ids[batch_idx]
            self.sequences_info[batch_idx]['updated_shifted_indices'][f'{timestep}'] = shifted_indices

        # Increase the `shifted_indices` to shift the vocab probs one step lower.
        # (note : `shifted_indices` will be used to indexing the `sorted_next_tokens_indices`)
        self.sequences_info[batch_idx]['updated_shifted_indices'][f'{timestep}'] += 1

        return reverted_input_ids, next_tokens

    def reverting_sequence(self, input_ids, next_tokens):
        """
        This function used to track if the reverting sequence has finished or not.
        """
        sequence = self.sequence_that_need_to_be_reverted[0]
        batch_idx = sequence['batch_idx']

        #  if it has finished the reverting process. Why + 1. This is because when the
        # `input_ids_when_detection_finished` get defined, the next_token wasn't get concatenated
        #  to the input_ids yet, while the algorithm expect it has.
        if (sequence['input_ids_when_detection_finished'].shape[1] + 1) == input_ids.shape[1]:
            # add this sequence back to untracked sequence
            self.untracked_sequence.append(sequence)
            
            self.sequence_that_need_to_be_reverted.pop(0)
            self.is_currently_reverting_sequence = False

            if len(self.sequence_that_need_to_be_reverted) != 0:
                self.start_revert = True

        # Store all the updated information during reverting process to the sequence
        timestep = input_ids.shape[1]
        if timestep not in self.sequences_info[batch_idx]['updated_timestep']:
            self.sequences_info[batch_idx]['updated_timestep'].append(timestep)
            self.sequences_info[batch_idx]['updated_next_tokens'][f'{timestep}'] = next_tokens[batch_idx]
            self.sequences_info[batch_idx]['updated_input_ids'][f'{timestep}'] = input_ids[batch_idx]
        else:
            self.sequences_info[batch_idx]['updated_next_tokens'][f'{timestep}'] = next_tokens[batch_idx]
            self.sequences_info[batch_idx]['updated_input_ids'][f'{timestep}'] = input_ids[batch_idx]

    def tracking(self, next_tokens, input_ids):
        """
        This function used to track the `next_tokens` that previously match to the
        first element of the ids in banned_words_ids.
        """
        if len(self.tracked_sequence) != 0:
            stop_looping_sequences = False
            for i, sequence in enumerate(self.tracked_sequence):

                banned_words_ids = sequence['detected_banned_words_ids']
                next_banned_word_idx = sequence['next_banned_words_idx']
                batch_idx = sequence['batch_idx']

                if sequence['pause_tracking'] == True:
                    if sequence['input_ids_when_paused'].shape[0] != input_ids[batch_idx].shape[0]:
                        continue
                    else:
                        if torch.all(sequence['input_ids_when_paused'].eq(input_ids[batch_idx])):
                            sequence['pause_tracking'] = False
                            sequence['input_ids_when_paused'] = None
                        else:
                            continue

                incomplete_ids = 0

                """
                In the following below, we will iterate over N number of ids.
                For any ids that match to the next_tokens first, we will revert
                this sequence immediately by executing prepare_to_revert() funct.
                """
                for ids in banned_words_ids:
                    if len(ids) == 1:
                        self.prepare_to_revert(input_ids, sequence)
                        stop_looping_sequences = True
                        break
                    else:

                        if next_tokens[batch_idx] == ids[next_banned_word_idx]:

                            # if we had reached to the final element of the ids /subset banned_words_ids
                            if next_banned_word_idx == (len(ids) - 1):
                                self.prepare_to_revert(input_ids, sequence)
                                stop_looping_sequences = True
                                break
                            else:
                                incomplete_ids += 1

                        else:
                            # if the next_tokens[batch_idx] doesnt match to the subsequent ids,
                            # we will remove this ids from the sequence.
                            self.tracked_sequence[i]['detected_banned_words_ids'].remove(ids)

                if stop_looping_sequences:
                    break

                if not self.start_revert:

                    # If none of the ids has completed yet we will increment the
                    # next_banned_words_idx by one.
                    if incomplete_ids == len(banned_words_ids):
                        self.tracked_sequence[i]['next_banned_words_idx'] += 1

                    # If none of the ids[next_banned_word_idx] match to the next_tokens[batch_idx],
                    # we will untrack this sequence.
                    if len(self.tracked_sequence[i]['detected_banned_words_ids']) == 0:
                        self.untrack_sequence(sequence)

        else:
            pass

    def detect(self, input_ids, next_tokens, sorted_next_tokens_indices):
        """
        This function will check the ``next_tokens`` with the first element of
        the self.banned_words_ids.
        """

        for ids in self.banned_words_ids:
            for sequence in self.untracked_sequence:
                # iterate over the untracked_sequence
                batch_idx = sequence['batch_idx']
                if next_tokens[batch_idx] == ids[0]:
                    # if the next_tokens of the sequence match to the first element of the ids
                    # we store all the necessary information for tracking purpose
                    sequence['input_ids_when_get_detected'] = input_ids
                    sequence['sorted_next_tokens_indices_when_get_detected'] = sorted_next_tokens_indices[batch_idx]
                    sequence['next_tokens_when_get_detected'] = next_tokens
                    sequence['detected_banned_words_ids'].append(ids)
                    sequence['next_banned_words_idx'] = 1

                    # Furthermore, we remove it from self.untracked_sequence and move it to
                    # self.tracked_sequence
                    self.untracked_sequence.remove(sequence)
                    self.tracked_sequence.append(sequence)

    def process(self, input_ids, next_tokens, sorted_next_tokens_indices):
        """
        This is the core function of this class. It get executed at every timestep
        receiving the input_ids, next_tokens that emitted from the generative model.

        Args:
            sorted_next_tokens_indices (:obj:`torch.LongTensor` of shape :obj:`(batch_size, vocab_size)`):
              The next_tokens_scores that has been sorted in descending order, with the highest score located at the first index
              and the lowest at the final index.
        """

        if self.is_currently_reverting_sequence:
            # step 4 in starbucks tissue paper
            """
            Please note at this timestep, the shifted next_token that emitted by
            the self.init_revert() function has been concatenated to the input_ids. 
            """
            self.reverting_sequence(input_ids, next_tokens)

        random_uniform = torch.rand((1,))
        if self.start_revert and self.epsilon > random_uniform:
            # step 1,2,3 in starbucks tissue papers
            input_ids, next_tokens = self.init_revert()
            self.is_currently_reverting_sequence = True
            self.start_revert = False

        timestep = input_ids.shape[1]
        is_input_ids_and_next_tokens_shifted_previously, updated_input_ids, shifted_next_tokens = self.does_the_input_ids_and_next_tokens_ever_being_shifted(
            input_ids, next_tokens, timestep)

        if is_input_ids_and_next_tokens_shifted_previously:
            # if the next_tokens at this timestep ever been shifted previously
            input_ids = updated_input_ids
            next_tokens = shifted_next_tokens

        self.tracking(next_tokens, input_ids)

        if not self.start_revert:
            self.detect(input_ids, next_tokens, sorted_next_tokens_indices)

        return input_ids, next_tokens