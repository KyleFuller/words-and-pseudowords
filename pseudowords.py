
import random as _random
from collections import abc as _abc
from collections import defaultdict as _defaultdict

def words_to_counts(words : _abc.Sequence[str], /) -> _abc.Mapping[str, _abc.Mapping[str, int]]:
    """
    Args:
        `words`: Contains strings that do not contain '\n'.
    
    Returns:
        Let chars be the set of all characters that occur in `words` together with '\n'.  Where augwords is `words` but with each word prepended and postpended with `\n`, the return value is a mapping from each char_1 in chars to a mapping from each char_2 in chars to a count of the number of times char_1 is followed by char_2 in augwords.  
        
        When neither char_1 nor char_2 are '\n', the count equals the number of times char_1 is followed by char_2 in words.  When char_1 is '\n', the count equals equals the number of times char_2 appears as the first character of a word in `words`.  When char_2 is '\n', the count equals the number of times char_1 appears as the last character of a word in `words`.
    """
    counts : _defaultdict[str, _defaultdict[str, int]] = _defaultdict(lambda: _defaultdict(lambda: 0))
    for word in [f'\n{word}\n' for word in words]:
        for c1, c2 in zip(word[:-1], word[1:]):
            counts[c1][c2] += 1
    return counts

def counts_to_markov_model(counts : _abc.Mapping[str, _abc.Mapping[str, int]], /) -> _abc.Callable[[str], str]:
    """
    Args:
        `counts`: a character transition count mapping provided by `words_to_counts`.
    
    Returns:
        the first order markov model resulting from `counts`.
    """
    def markov_model(c1 : str) -> str:
        c2s = list(counts[c1].keys())
        c2_counts = [counts[c1][c2] for c2 in c2s]
        return _random.sample(c2s, 1, counts=c2_counts)[0]

    return markov_model

def markov_model_to_pseudoword_generator(markov_model : _abc.Callable[[str], str], /) -> _abc.Generator[str]:
    """
    Args:
        `markov_model`: a markov model provided by `counts_to_markov_model`.
    
    Returns:
        An infinite generator of pseudowords for which the starting character is produced by `markov_model('\n')`, each next character when applying the Markov model to the current character does not result in '\n' is the Markov model's output, and the string ends when the applying the Markov model to the current character results in '\n'.
    """
    while True:
        cur = '\n'
        characters : list[str] = []
        while True:
            cur = markov_model(cur)
            if cur == '\n':
                yield ''.join(characters)
                break
            characters.append(cur)

def words_to_pseudoword_generator(words : _abc.Sequence[str], /) -> _abc.Generator[str]:

    """ 
    Args:
        `words`: Contains strings that do not contain '\n'.

    Returns: 
        `markov_model_to_pseudoword_generator(counts_to_markov_model(words_to_counts(words)))`
    """

    return markov_model_to_pseudoword_generator(counts_to_markov_model(words_to_counts(words)))