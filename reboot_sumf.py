from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import speech_recognition as sr
from os import path
from pydub import AudioSegment

# convert mp3 file to wav                                                       
sound = AudioSegment.from_mp3("sample.mp3")
sound.export("transcript.wav", format="wav")


# transcribe audio file                                                         
AUDIO_FILE = "transcript.wav"

# use the audio file as the audio source                                        
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file                  

text_str=''' Lots of kids are very active. But when it’s time to stop and settle down, they typically manage to put on the brakes. They might kick a ball around one minute and sit quietly with a book five minutes later.But some kids just can’t keep still. They’re forever fidgeting, grabbing things, talking, or running even after they’re told to stop. They’re more than just active. Experts would describe them as hyperactive.Kids don’t act like this on purpose. They have a need to keep moving and haven’t yet developed the skills to manage it.Some people see hyperactive kids and make judgments. They might think it’s a discipline problem or that a child’s just being rude. They might also make comments that leave kids (or you) feeling bad and ashamed.If your child is in constant motion, you may wonder about the behavior you’re seeing. Learn more about hyperactivity in kids.Hyperactive Behavior You Might Be Seeing What is hyperactivity? Some people think it’s just kids racing around all the time. But there’s much more to it.Hyperactivity is constantly being active in ways that aren’t appropriate for the time or setting. It’s the constant part that makes the big difference. If it happened once or twice, nobody would think much of it.Here are a few examples of what hyperactive kids might frequently do:Run and shout when playing, even if they’re indoors Stand up in the middle of class and walk around while the teacher talks Move so fast they bump into people and things Play too roughly and accidentally hurt kids or themselves Hyperactivity can look different at different ages, and it can vary from child to child. Here are some of the behaviors you might see, beyond running and jumping around: Seems to talk constantly Frequently interrupts others Moves from place to place quickly and often clumsily Keeps moving even when sitting down Bumps into things Fidgets and has an urge to pick up everything and play with it Has trouble sitting still for meals and other quiet activities What Can Cause Hyperactivity
Hyperactivity isn’t the same thing as being very active. It’s constant and beyond kids’ control. Kids aren’t hyperactive because of a lack of discipline or because they’re defiant. In fact, kids who are overactive often want to settle down so it’s easier to join in what’s going on. It can be really frustrating to have trouble doing what they know is expected of them.One thing to consider with hyperactivity is age. It takes time for kids to develop the self-regulation skills they need to keep their behavior in check. Also, kids don’t all develop at the same rate. One child might have good self-control at age 4 while it takes another child until age 6.But there comes a point where most kids in an age group have similar self-regulation skills. That’s when it’s often clearer if kids are lagging.
One of the main causes of hyperactivity is ADHD, a common condition that results from differences in the brain.Hyperactivity is a core symptom of ADHD. ADHD doesn’t go away as kids get older, but the hyperactivity piece often does, or it becomes less extreme. That often happens in adolescence. (Read more about hyperactivity in teens.)There are also certain medical and mental health conditions that can cause hyperactive behavior. Thyroid issues, lack of sleep, anxiety, and mental distress related to things like abuse can all lead to hyperactivity. Starting puberty can cause kids to be hyperactive, too.
What Can Help With HyperactivityLook for patterns in your child’s behavior. When does your child seem most hyperactive? What does the hyperactivity look like? For instance, maybe it shows up as restlessness, fidgeting, or constant talking. Knowing these patterns will help you to be specific if you talk to your child’s doctor or teacher.
Your child’s teacher is great source for gathering information. Connect with the teacher to get a sense of what’s happening in the classroom. You can also find out if the teacher has any tips you can try at home. For instance, maybe the teacher offers brain breaks during class or lets your child use a fidget.
Give your child plenty of ways to stay active through games, sports, physical chores, and activities. You can even try apps to help kids build self-control. (Check out apps for younger kids and for teens and tweens.)If your child has trouble settling down for homework or dinner, find a repetitive activity for your child to do for five to 10 minutes before starting. Word searches, crosswords, jigsaw puzzles, and card games are good for this.And if you think your child might have ADHD, there are steps you can take to find out. A good first step is to get to know other signs of ADHD, and take notes on what you see.Hyperactivity can be hard on the whole family (as well as teachers). One important thing you can do is to help your child not feel bad or ashamed. Assure your child it’s common, and that hyperactivity will get better with time and support. '''


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.0 * threshold)

    return summary


if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)
