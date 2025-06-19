import dataclasses
from typing import List, Tuple



## Special Tokens
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_HIDDEN_TOKEN = "<HIDDEN>"
DEFAULT_PRE_TOKEN = "<PRE>"
DEFAULT_START_TOKEN = "<START>"
DEFAULT_END_TOKEN = "<END>"

@dataclasses.dataclass
class prompt:
    hidden_token: str = DEFAULT_HIDDEN_TOKEN
    start_token: str = DEFAULT_START_TOKEN
    end_token: str = DEFAULT_END_TOKEN
    pre_token: str = DEFAULT_PRE_TOKEN

    def get_prompt_question(self,home_loc_id, traj_length):
        prompt = (f"Given user's hidden state {self.start_token} {self.hidden_token} {self.end_token}  and home location {self.start_token} {home_loc_id} {self.end_token}, please generate the whole mobility\
                trajectory with length {traj_length} of the user.")
        return prompt

    def get_prompt_response(self, traj: list) -> str:
        """
        Format a trajectory list into a response string.
        Args:
            traj (list): A list of location identifiers (e.g., [101, 102, 103]).
        Returns:
            str: A string like " <Answer> 101 102 103"
        """
        response = self.pre_token + ':' + '[' + ' '.join(str(loc) for loc in traj) + ']'
        return response

    def get_prompt_train(self, home_loc_id: int, traj: list,traj_length: int) -> str:
        """
        Construct a full prompt by combining a question (based on home location)
        and a formatted response from the given trajectory.
        Args:
            home_loc_id (int): The ID of the user's home location.
            traj (list): A list of location identifiers, e.g., [101, 202, 303].
        Returns:
            str: The complete prompt string to be used as input.
        """
        question = self.get_prompt_question(home_loc_id, traj_length)
        response = self.get_prompt_response(traj)
        return question + response
    def get_prompt_eval(self, home_loc_id: int, traj_length: int) -> str:
        """
        Construct a full prompt by combining a question (based on home location)
        and a formatted response from the given trajectory.
        Args:
            home_loc_id (int): The ID of the user's home location.
            traj (list): A list of location identifiers, e.g., [101, 202, 303].
        Returns:
            str: The complete prompt string to be used as input.
        """
        question = self.get_prompt_question(home_loc_id, traj_length)
        question+= self.pre_token + ':'
        return question

