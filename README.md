# Word-Pseudoword Classifier using Convolutional Neural Network

In this demonstration, a Convolutional Neural Network is trained to distinguish real English words from “pseudowords”, achieving ~ 89.5% testing accuracy.  The real words are taken from a list of 194,000 words.   The pseudowords are generated using a character-level Markov model whose transition probabilities are derived from the transition frequencies found in the same word list.  To limit the size of the network, all words (both real and pseudo) are filtered to be at most 12 characters in length.  A few pseudowords are shown here:

```
riedring
angor
coy
lc
bandenjes
fliath
ustrades
bialomisis
tilico
chtiomerapis
```

One can see that "coy" in the above list is infact a real word as well.  As such, the goal of the classification task is not to classify a word according to some intrinsic property of “realness”, but rather to classify an *instance* of a word according to its *source* — either the word bank or the Markov model.  This means it is impossible to achieve 100% accuracy.  Nonetheless, we can get close as is demonstrated here.
 
The most straightforward model one might try is a perceptron that takes 12 * 27 features (representing 12 characters that each could be one of the letters of the alphabet or a “padding” that occurs after the ends of words) and simply takes a linear combination of these features.  This approach appears to top out at only around 66% accuracy.  One might wonder why a perceptron was able to perform even *that* well, rather than being as good as a coin flip.  After all, the character and digraph frequencies in pseudowords are very similar to those in real words.  This is likely because the perceptron is given the *count* of each character in a word, but the *sequence* of characters, which is more informative.  To illustrate, suppose the Markov model starts a word off with ‘n’, a fairly common starting letter among real words.  In English, ‘n’ is often followed by ‘g’ (e.g. in “running”), so the Markov model might continue with ‘g’, resulting in ‘g’ being a fairly common second letter among pseudowords (roughly 1.33% based on my testing).  However, ‘g’ is not a very common second letter among real words (roughly 0.33% based on the 194000 real words), so a ‘g’ in the second position gives the perceptron *some* evidence that the word is a pseudoword.  Of course, we can do much better than 66% accuracy.

A more effective model for this task is a dense feedforward network that takes the same features as the perceptron but has hidden layers.  With some playing around, one hidden layer with 648 neurons appears to work relatively well, yielding a testing accuracy of ~ 86%.  Unlike the perceptron, this network can exploit nonlinear relationships, resulting in much better accuracy.

However, a limitation here is that if there is some pattern that suggests that a word is a real word or pseudoword wherever it may appear in a word, this network has to relearn the importance of that pattern in every position where it appears.  This suggests that the feedforward network still makes suboptimal use of its training data, and also suggests a possible solution.  By using a convolutional neural network (CNN) instead,  we can learn to look for *features* regardless of where they appear in the word.  With a CNN with 200 features in the first hidden layer, 200 in the second, and then a dense layer of 50 neurons after global max pooling, we get ~ 89.5% accuracy.  

In the future, an interesting thing to try might be to make the task harder by using a Markov model that takes into account not just the identity of the previous character, but its position in the word.  That is, it would consider the states to be not characters but (character, position) pairs.  We saw that with the current Markov model, a perceptron is able to achieve 66% accuracy by using position information, but with a Markov model that takes into account position, the perceptron would likely do no batter than chance.

The idea for this demonstration is partly inspired by a [video](https://www.youtube.com/watch?v=evTx5BoKcc8) in which Cary Huang uses a dense feedforward neural network (not a recurrent neural network as the title mistakenly suggests) to classify words as either English or Mandarin.  The use of a CNN here is my innovation, and I was pleasantly surprised to see how much it helped.  In an effort to get as much out of this demonstration as possible, I wrote all of the code and documentation by hand.

Note: The word list 194000.txt can be downloaded from http://www.gwicks.net/textlists/english3.zip and is subject to JUST WORDS! licensing terms.
