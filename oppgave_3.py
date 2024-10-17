import matplotlib.pyplot as plt
import numpy as np
from utils import onehot
from layers import *
from data_generators_corrected import get_train_test_sorting
from data_generators_corrected import get_train_test_sorting

# Oppgave 3.2 - Algoritme 4, trening av nevralt nettverk i batcher

def trening_av_nevralt_nettverk(neural, loss, x, y, m, n_batches, n_iter, alpha=0.01, beta1 = 0.9, beta2 = 0.999):
    """
    neural: Nevralt nettverk
    loss: Objektfunksjon (=Crossentropy())
    x, y: Dattasett vi tester på
    m: antall siffer vi
    n_iter: Antall iterasjoner som skal gjennomføres
    alpha, beta1, beta2: Parametre til adam_step
    """

    Loss_list = []  # liste for å lagre gjennomsnittlig objektfunksjon over batchene for hver iterasjon

    for j in range(n_iter):  # for hver iterasjon

        total_loss = []      # legger til losset for hver batch for å kunne regne gjennomsnittet

        for b in range(n_batches): 

            X = onehot(x[b], m)
            Y_hatt = neural.forward(X)         # beregner Y_hat ved å gjennomføre et forward pass

            L_jk = loss.forward(Y_hatt, y[b])  # beregner loss for denne iterasjonen
            total_loss.append(L_jk)

            dLdZ = loss.backward()          # beregner gradienten med backward pass i loss-funksjonen
            neural.backward(dLdZ)           # kjører et backward pass

            neural.adam_step(alpha, beta1, beta2)   # Adam oppdatering for hver parameter

        Loss_list.append(np.mean(total_loss))  # legg til gjennomsnittlig objektfunksjon for denne iterasjonen

    
    # Ploter Loss_list med logaritmisk skala på y-aksen:
    plt.plot(np.arange(n_iter), Loss_list)
    plt.yscale('log')
    plt.xlabel('Iterasjon')
    plt.ylabel('Gjennomsnittlig objektfunksjon')
    plt.title('Objektfunksjon over iterasjoner')
    plt.show()

    return neural


#Oppgave 3.3 sortering av data

def sorting_data(D, m, neural, r):

    '''
    Sorting_data er en funksjon som tar inn datasett laget av data_generators_corrected og bruker
    testdata til å gjøre prediksjoner på å sortere sekvenser av tall og gir ut andel av 
    suksessfulle prediksjoner.
    Input:
    D - datasett
    m - antall siffere programmet skal gjette på, fra 0 til og med m-1
    neural - det nevrale nettverket som blir trent
    r - lengden på sekvensen som skal predikeres
    Output:
    Andel prediksjoner som var korrekte.
    '''

    x_test = D['x_test']    #tar ut testdata for x og y fra datasettene

    y_test = D['y_test']

    f, g, h = np.shape(y_test)  #finner shapen til y_test som skal predikeres

    x = x_test[0]   #x_test kommer ut som tredimensjonal, siden det bare er en batch definerer
                    #vi denne som x for enklere utregning

    for i in range(r):  #iterer gjennom nok ganger til å predikere alle siffere

        X = onehot(x, m)

        z = neural.forward(X)

        z_hat = np.argmax(z, axis = 1)[:,-1]    #tar siste kolonnen fra prediksjonen

        x = np.column_stack((x,z_hat))  #setter denne kolonnen på x slik at den kan brukes i 
                                        #de videre prediksjonene
   
    antall_suksess = 0  #definerer antall suksesser

    y_hatt = x[:, -h:]  #tar ut de siste sifferne i prediksjonen for å sammenligne med y_test

    for i in range(g):  #itererer gjennom hvert datasett

        if np.array_equal(y_hatt[i], y_test[0][i]): #sjekker om hvert siffer i arrayet er likt

            antall_suksess += 1       

    return antall_suksess / g       #returnerer andel ganger prediksjonen var riktig


#oppgave 3.4 addisjon av data
 
def adding_data(D, m, neural, r):

    """
    Adding_data er en funksjon som tar inn datasett laget av data_generators_corrected og bruker
    testdata til å gjøre prediksjoner på summen av to to-sifret tall og gir ut andel av 
    suksessfulle prediksjoner.
    Input:
    D - datasett
    m - antall siffere programmet skal gjette på, fra 0 til og med m-1
    neural - det nevrale nettverket som blir trent
    r - antall siffere i tallene som summeres
    Output:
    Andel prediksjoner som var korrekte.
    """

    x_test = D['x_test']    #tar ut test dataen for x og y fra datasettet

    y_test = D['y_test']

    f, g, h = np.shape(y_test)    #finner dimensjonene til y_test

    x = x_test[0]       #tar ut første og eneste batch i x_test

    for i in range(r+1):    #iterer antall ganger lik antall siffere i tallene som summes + 1 

        X = onehot(x, m)

        z = neural.forward(X)

        z_hat = np.argmax(z, axis = 1)[:,-1]    #tar ut siste kolonne fra prediksjonen

        x = np.column_stack((x,z_hat))   #setter på kolonnen til x slik at den kan brukes
                                         #til de videre prediksjonene

    antall_suksess = 0     #definerer antall suksesser

    y_hatt = x[:, -h:]     #tar de siste kolonnene i prediksjonen lik antall kolonner i y_test

    for i in range(g):  #itererer gjennom hvert datasett

        if np.array_equal(y_hatt[i,::-1], y_test[0,i]):   #sammenligner y_test med prediksjonen
                                                          #den reverseres siden koden predikerer
            antall_suksess += 1                           #fra bakerst og forover

    return antall_suksess / g    #returner antall korrekte prediksjoner


def alle_tosifrede_tall():
    """
    Genererer predikerte løsninger for alle 10 000 tallparene. 
    """
    
    #Lager x- og y-arrays med dimensjon henholdsvis (100, 100, 4) og (100, 100, 3) med alle de mulige tallparene
    x_verdier = np.zeros((100, 100, 4), dtype=int)
    y_verdier = np.zeros((100, 100, 3), dtype=int)
    
    for i in range(100):
        
        for j in range(100):
            
            x_verdier[i,j] = [i // 10, i % 10, j // 10, j % 10]  
            
            sum_tall = i+j
            
            if sum_tall < 100:   # dersom summen blir tosifret legges det til en 0, slik at også disse tallene blir tresifret
                
                y_verdier[i,j] = [0, sum_tall // 10, sum_tall % 10]  
                
            else:
                
                y_verdier[i,j] = [sum_tall // 100, sum_tall // 10 % 10, sum_tall % 10] 
    
    D = {"x_test": x_verdier, "y_test": y_verdier}

    return D