import os
import re
import unicodedata
import random
import numpy as np
import matplotlib.pyplot as plt

spam_counter = 0
ham_counter = 0
spam_classifier = 0.5
stop_word_classifier = 400
laplace = 1
classes = 0

class Email:
    def __init__(self, words, isSpam, googleCsfc):
        self.words = words
        self.isSpam = isSpam
        self.googleCsfc = googleCsfc


## ------------------------- UTILITÁRIOS ------------------------- ##

# Lê os emails da database

def getData():

    email_path = os.getcwd() + '\data\mail'
    spam_path = os.getcwd() + '\data\spam'

    email_list = read(email_path, False)
    spam_list = read(spam_path, True)

    emails = email_list + spam_list

    global spam_counter, ham_counter
    spam_counter = len(spam_list)
    ham_counter = len(email_list)
    return (emails)


# Recebe um diretório e lê todos seus arquivos

def read(path, isSpam):
    emails = []
    for txt in os.listdir(path):
        try:
            with open(os.path.join(path, txt), "r", encoding='ISO-8859-1') as file:
                line = file.read()
                gClassification = line[0]
                line = line[1:]
                line = normalize(line)
                line = line.split(' ')
                email = Email(line, isSpam, gClassification)
                emails.append(email)
        except Exception as e:
            print(e)

    return emails


# Normaliza o texto, o tornando lower case e removendo acentos,
# caracteres especiais e multiplos espaços

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = text.replace('\n', ' ')
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub("\s\s+" , " ", text)
    return text


# Divide os sets de treinamento e de teste

def split_sets(email_list): 
    selecteds = int(len(email_list)*0.8)
    random.shuffle(email_list)
    return (email_list[:selecteds], email_list[selecteds:])


# Gera a lista de stop-words, as palavra mais frequentes no conjunto de treinamento

def getStopWords(word_collection):
    stop_words = []

    all_words = {}

    for word in word_collection:
        total =  word_collection[word]["total"]
        spam_freq = word_collection[word]["spam"]
        if total >= stop_word_classifier and not isBiased(total, spam_freq):
            stop_words.append(word)

        all_words[word] = total

    generateGraph(all_words)    
    return stop_words


# Verifica se a frequência da palavra é alta apenas em SPAM e em não-SPAM, apenas
# considera stop-word se for frequente nos dois conjuntos de forma menos tendenciosa

def isBiased(total, spam_freq):
   
    frequency = spam_freq/total
    if frequency >= 0.3 and frequency <= 0.7:
        return True
    else:
        return False

# Plota um gráfico com frequência de cada palavra

def generateGraph(words):

    total = 0 
    for key, value in words.items():
        total += value
    
    avg = total/len(words)
    
    x = []
    y = []
    
    for key, value in words.items():
        if value >= avg*10:
            x.append(value)
            y.append(key)
   
    plt.bar(y, x, align='center', alpha=0.5)
    plt.xticks(rotation='vertical')
    plt.ylabel('Frequency')
    plt.title('Words frequency')

    plt.show()


# Remove as stop-words do conjunto de teste

def removeStopWords(testing, stop_words):

    for email in testing:
        email.words = [word for word in email.words if word not in stop_words]


## ------------------------- TREINAMENTO ------------------------- ##

# Cria um dicionário de cada palavra contida nos emails
# terá 3 atributos:
# "total" = total de vezes que a palavra apareceu
# "spam" = número de vezes que a palavra apareceu em um spam
# "probability" = teorema de Bayes: P(S | W) = ( P(W | S) * P (S) ) / (( P(W | S) * P (S) ) ( P(W | NOT S) * P (NOT S) ))

def training(emails):
    word_collection = {}
    
    for email in emails:
        
        for word in email.words:
         
            if word in word_collection:

                if email.isSpam:
                    word_collection[word]["spam"] += 1
                
                word_collection[word]["total"] += 1

            else:

                if email.isSpam:
                    word_collection[word] = {"total": 1, "spam": 1}
                else:
                    word_collection[word] = {"total": 1, "spam": 0}

    global classes
    classes = len(word_collection.keys())
    
    for email in emails:
        for word in email.words:
            setProbabilities(word_collection, word)

    

    return word_collection


# Usa o teorema de Bayes para calcular a chance de uma mensagem ser spam, se tal palavra estiver nela
# P(S | W) = ( P(W | S) * P (S) ) / (( P(W | S) * P (S) ) ( P(W | NOT S) * P (NOT S) ))
# P(S | W) = result 
# P(W | S) = prob_word_in_spam
# P(S) = prob_spam
# P(W | NOT S) = prob_word_in_ham
# P(NOT S) = prob_ham

def setProbabilities(word_collection, word):
   
    prob_word_in_spam = word_collection[word]["spam"] / word_collection[word]["total"]
    prob_word_in_ham = 1 - prob_word_in_spam
   
    total_emails = (spam_counter + ham_counter)
    prob_spam = spam_counter / total_emails
    prob_ham = ham_counter / total_emails

    result = ((prob_word_in_spam * prob_spam)) / ((prob_word_in_spam * prob_spam) + (prob_word_in_ham * prob_ham))

    result += laplace/classes
    word_collection[word]["probability"] = result


## ------------------------- TESTE E AVALIAÇÃO ------------------------- ##

# Recebe o conjunto de teste e classifica cada email contido nele

def test(word_collection, emails):
    
    score = 0
    gScore = 0
    for email in emails:
        p = []
        for word in email.words:
            
            if word in word_collection:
                p.append(word_collection[word]["probability"])
            else:
                probability = laplace / classes
                p.append(probability)
            

        if classify(p) == email.isSpam:
            score += 1
        gScore += email.googleCsfc
    accuracy = score/len(emails)
    gAccuracy = gScore/len(emails)
    return accuracy, gAccuracy


# Define a probabilidade de um email ser spam, dada a probabilidade de suas palavras serem spam
# P(S) = P(S | W1) * ... * P(S | Wi) / ( P(S | W1) * ... * P(S | Wi) +  P(NOT S| W1) * ... * P(NOT S | Wi) )

def classify(prob):
    inv = [(1 - x) for x in prob]
    prob_mult = np.prod(prob)
    prob_mult_inv = np.prod(inv)
  
    result = prob_mult / (prob_mult + prob_mult_inv)

    if result >= spam_classifier:
        return True
    else: 
        return False
    
def main():

    (email_list) = getData()
    (training_set, testing_set) = split_sets(email_list)
   
    word_collection = training(training_set)
    accuracy, gAccuracy = test(word_collection, testing_set)
    print('Acurácia inicial: ', accuracy)
    print('Acurácia Gmail: ', gAccuracy)

    stop_words = getStopWords(word_collection)
    removeStopWords(testing_set, stop_words)
    accuracy, gAccuracy = test(word_collection, testing_set)
    print('Acurácia (sem stop-words): ', accuracy)
    print('Acurácia Gmail: ', gAccuracy)


main()