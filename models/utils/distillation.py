"""
Model Distillation
    https://towardsdatascience.com/distillation-of-bert-like-models-the-theory-32e19a02641f
    https://towardsdatascience.com/distillation-of-bert-like-models-the-code-73c31e8c2b0a
"""
from typing import Any, Tuple

import torch
from torc


h import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaModel, RobertaPreTrainedModel, RobertaConfig


def visualize_children(object: Any, level: int = 0,) -> None:
    """
    Prints the children of (object) and their children too, if there are any.
    Uses the current depth (level) to print things in a ordonnate manner.
    """
    print(f"{'   ' * level}{level}- {type(object).__name__}")
    try:
        for child in object.children():
            visualize_children(child, level + 1)
    except:
        pass


def distill_model(teacher_model: RobertaPreTrainedModel) -> RobertaPreTrainedModel:
    """
    Distilate a teacher_model like DistilBERT for a BERT model.
    The student model has the same configuration, except for the number of hidden layers, which is // by 2.
    The student layers are initilized by copying one out of two layers of the teacher, starting with layer 0.
    The head of the teacher is also copied.
    """
    # Get teacher configuration as a dictionnary
    model_config = teacher_model.config.to_dict()

    # Half the number of hidden layer
    model_config['num_hidden_layers'] //= 2
    
    # Convert the dictionnary to the student configuration
    model_config = RobertaConfig.from_dict(model_config)
    
    # Create uninitialized student model
    student_model = type(teacher_model)(model_config)

    # Initialize the student's weights
    distill_model_weights(teacher=teacher_model, student=student_model)

    # Return the student model
    return student_model


def distill_model_weights(teacher: nn.Module, student: nn.Module,) -> None:
    """
    Recursively copies the weights of the teacher to the student.
    This function is meant to be first called on a AutoModel, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """
    # If the part is an entire RoBERTa model or a RobertaFor..., unpack and iterate
    if isinstance(teacher, RobertaModel) or type(teacher).__name__.startswith('RobertaFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_model_weights(teacher_part, student_part)

    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, RobertaEncoder):
            teacher_encoding_layers = [layer for layer in next(teacher.children())]
            student_encoding_layers = [layer for layer in next(student.children())]
            for i in range(len(student_encoding_layers)):
                student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    
    # Else the part is a head or something else, copy the state_dict
    else:
        student.load_state_dict(teacher.state_dict())


class ModelDistillator(nn.Module):
    """
    A class to distillate a BERT-like model.
    """
    def __init__(self,teacher_model: RobertaPreTrainedModel, temperature: float = 1.,) -> None:
        """
        Initiates the Distillator with the (teacher_model) to distillate from.
        """
        super(ModelDistillator, self).__init__()
        self.teacher = teacher_model
        self.student = distill_model(teacher_model)
        self.temperature = temperature

    @property
    def temperature(self) -> float:
        """
        The temperature used for training can change, but for inference it is always 1.
        """
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(self, value : float,) -> None:
        """
        The temperature must always be above 1. Otherwise, an error is raised.
        """
        if value < 1:
            raise(ValueError(f"Temperature must be above 1, it cannot be {value}"))
        self._temperature = value

    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, from_teacher: bool = False,) -> torch.Tensor:
        """
        Given a couple of (input_ids) and (attention_mask), returns the logits corresponding to the prediction.
        The logits come from the student unless (from_teacher) is set to True, then it's from the teacher.
        """
        if from_teacher:
            return self.teacher.classifier(self.roberta(input_ids, attention_mask)[0])
        return self.student.classifier(self.student(input_ids, attention_mask)[0])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a couple of (input_ids) and (attention_mask), returns the logits corresponding to the prediction.
        Also takes in the (labels) associated to the inputs.
        Returns the student probability distibution with temperature 1 and the loss.
        """
        student_logits = self.get_logits(input_ids, attention_mask, False)
        teacher_logits = self.get_logits(input_ids, attention_mask, True)
        return student_logits.softmax(1), self.loss(teacher_logits, student_logits, labels)

    def loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor, labels: torch.Tensor, ) -> torch.Tensor:
        """
        The distillation loss for distilating a BERT-like model.
        The loss takes the (teacher_logits), (student_logits) and (labels) for various losses.
        """
        # Temperature and sotfmax: 
        #       https://miro.medium.com/max/1400/1*oRqw7kWld7qvdkkQjvsyzQ.png
        student_logits = (student_logits / self.temperature).softmax(1)
        teacher_logits = (teacher_logits / self.temperature).softmax(1)

        # Classification loss (problem-specific loss)
        loss1 = CrossEntropyLoss()(student_logits, labels)

        # CrossEntropy teacher-student loss
        loss2 = CrossEntropyLoss()(student_logits, teacher_logits)

        # Cosine loss
        loss3 = CosineEmbeddingLoss()(teacher_logits, student_logits, torch.ones(teacher_logits.size()[0]))
        
        # Average the loss and return it
        loss = (loss1 + loss2 + loss3) / 3
        return loss

