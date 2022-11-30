# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:59:49 2022

@author: USER
"""
#%% required libraries
import numpy as np
import random
#%% Initial card and drawing funciton
def initial():
    dsum=np.random.randint(1,11)
    psum=np.random.randint(1,11)
    return dsum, psum 
def draw():
    card= np.random.randint(1,11)
    if np.random.random()<=1/3:
        return -card
    else:
        return card
#%% The step function
#s is a 2x1 vector with 2 values: the dealer's first card and the player's sum
#a is the player's action given state s 
# if the player sticks(a=0), it's the dealer's turn to draw cards. Rewards are calculated accordingly
# if the player hits(a=1), the player draws a card and his sum is updated
def step(dsum,psum,a):
    if a==0:
       terminated=True
       while 0<dsum<17:
           dsum+= draw()
       if dsum>psum:
           reward=-1
       elif dsum==psum:
           reward=0
       elif not(0<dsum<=21) or psum>dsum:
           reward=1    
    elif a==1:
        psum+=draw()
        if not (0<psum<=21):
            reward=-1
            terminated=True
        else:
            reward=0
            terminated=False
    
    return dsum,psum,reward,terminated  

#%% Monte Carlo Method
#learning rate = 1/N(s,a)
#epsilon rate=100/(100+N(s))
#epsilon greedy exploration strategy

#Q value update rule 
def update(N,G,Q,state_actions): 
    for i in state_actions:
        Q[i]=Q[i]+(1/N[i])*(G-Q[i])
    return Q 

#Q function initialization. 10 dealer hand values, 21 player sum values, 2 actions
Q=np.zeros((10,21,2),dtype='float')
#State-action visit counter initialization.
N=np.zeros((10,21,2),dtype='int')
#Policy Initiation. It is a function 
def policy(dsum,psum,Q,N):
    epsilon=100/(100+np.sum(N[dsum-1,psum-1,:]))
    greed=np.random.random()
    if greed>epsilon:
        a=np.argmax(Q[dsum-1,psum-1,:])
    else:
        a=random.choice([0,1])
    return a
#Episode Generation 
for i in range(100000):
    visited=[]
    current_state=initial()
    G=0
    end=False 
    while end==False: 
        #set the action taken 
        action=policy(current_state[0],current_state[1],Q,N)
        #update visited states 
        visited.append((current_state[0]-1,current_state[1]-1,action))
        N[current_state[0]-1,current_state[1]-1,action]+=1
        #move to the next state 
        temp=step(current_state[0],current_state[1],action)
        current_state=(temp[0],temp[1])
        end=temp[3]
        G+=temp[2]
    Q=update(N,G,Q,visited)

#VISUALIZATION NEEDED 
#%% Temporal Difference Method
#Initialize loop variables and tables for the two plots 
episodes=1000
lambda_matrix=np.arange(0,1.1,0.1).round(1)
#tables for the evolution of mse for while running episodes. mse per episode has size the number of λ values*number of episodes
mse_per_episode=np.zeros((len(lambda_matrix),episodes))
#mse per lambda indicates the "final mse" obtained after 1000 episodes for each lambda. This chart can show us which lamda is optimal and then we can draw inference.
mse_per_lambda=np.zeros(len(lambda_matrix))

for i,λ in enumerate(lambda_matrix):
    #Q function initialization

    tdq=np.zeros((10,21,2),dtype='float')
    
    #State-action visit counter initialization for epsilon update
    tdn=np.zeros((10,21,2),dtype='int')    
    
    
    #Episode Initialization
    for game in range(episodes):
        #Initialize eligibility vector
        E=np.zeros((10,21,2),dtype='float')
        current_state_TD=initial()
        #set the action taken 
        action=policy(current_state_TD[0],current_state_TD[1],tdq,tdn)
        #first state guaranteed to be non terminal by environment limitations
        terminal=False
        while terminal==False:
            #At this point, there was an error with the definition of next_state where step was called 2 times and the terminal variable received different results incompatible with the states
            next_state=step(current_state_TD[0],current_state_TD[1],action)
            reward=next_state[2] 
            terminal=next_state[3]
            print(next_state)
            # update N 
            tdn[current_state_TD[0]-1,current_state_TD[1]-1,action]+=1  
            #make td error according to state terminality 
            if terminal==True:
                td_error=reward-tdq[current_state_TD[0]-1,current_state_TD[1]-1,action]
            else:
                #observe next action and reward 
                next_action=policy(next_state[0],next_state[1],tdq,tdn)
                td_error=reward+tdq[next_state[0]-1,next_state[1]-1,next_action]-tdq[current_state_TD[0]-1,current_state_TD[1]-1,action]
            
            E[current_state_TD[0]-1,current_state_TD[1]-1,action]+=1
            #update rule
            alpha=1/ tdn[current_state_TD[0]-1,current_state_TD[1]-1,action]
            tdq=tdq+E*td_error*λ*alpha
            E=E*λ
            #Make the next state the current state if it is not terminal
            if terminal == False:
                current_state_TD=next_state
                action=next_action
        Δ=tdq-Q
        #denominator is total of state action pairs, which is 21*10*2=420
        mse=np.sum(np.square(Δ))/420
        mse_per_episode[i,game]= mse
    mse_per_lambda[i]= mse
#SARSA uses eligibility traces which assign credit/balance to each state-action pair. E stands for the eligibility matrix
#It weighs the update of Q(s,a) towards our latest actions

#%% Linear Funciton Approximation
#I am implementing an algorithm similar to the SARSA algorithm above. The 3 key differences are: Q is replaced with L, a binary feature vector multiplied by weights. Also, the epslion and the learning rate are both given and constant. 
#episode and lambda already defined 
#new mse per episdoe and mse per lambda matrices 
mse_per_episode_LFA=np.zeros((len(lambda_matrix),episodes))
mse_per_lambda_LFA=np.zeros(len(lambda_matrix))
#define new policy for different epsilon 
def policy_lfa(dsum,psum,Q):
    epsilon=0.05
    greed=np.random.random()
    if greed>epsilon:
        a=np.argmax(Q[dsum-1,psum-1,:])
    else:
        a=random.choice([0,1])
    return a
#define φ, the feature vector
def φ(s,a): 
    φ=np.zeros((3,6,2)) 
    for dealer_i,dealer_interval in enumerate(features['dealer']):
        if dealer_interval[0] <=s[0]<= dealer_interval[1]:
                for player_i, player_interval in enumerate(features['player']):
                    if player_interval[0] <= s[1] <= player_interval[1]:
                        φ[dealer_i,player_i,a]=1
        return φ
#Initialize feature space
features={'dealer':[[1,4], [4,7],[7,10]],  'player':[[1, 6],[4, 9],[7,12],[10, 15],[13, 18],[16, 21]], 'action':[0,1]}
#For every λ...
for i,λ in enumerate(lambda_matrix):
    #Q function initialization
    lq=np.zeros((10,21,2),dtype='float')
    #Initialize weight vector
    w = np.zeros((3,6,2))
    #Generate episodes
    for game in range(episodes):
        #Initalize eligibility trace for the weight vector
        ew = np.zeros((3,6,2))
        current_state_lfa=initial()
        #set the action taken 
        action=policy_lfa(current_state_lfa[0],current_state_lfa[1],lq)
        #first state guaranteed to be non terminal by environment limitations
        terminal=False
        while terminal==False:  
            next_state=step(current_state_lfa[0],current_state_lfa[1],action)
            reward=next_state[2]
            terminal=next_state[3]
            #call φ
            f=φ(current_state_lfa,action)
            #Add to eligibility traces
            ew+=f
            #make td error according to state terminality. The following formula is for when terminal==True. If false it is adjusted
            td_error=reward-np.sum(w*f)
            if terminal==False:
                #observe next action and reward 
                next_action=policy_lfa(next_state[0],next_state[1],lq)
                f=φ(next_state,next_action)
                td_error+=np.sum(w*f)
            #weight and eligibility trace update rule
            alpha=0.01
            w+=ew*td_error*alpha
            ew=ew*λ
            #Make next s,a current s,a for the next iteration
            current_state_lfa=next_state
            #Check if next state is terminal for action update. Terminal was update in the previous for loop for the next state
            if terminal==False:
                action=next_action
        #Q update
        for d in range(10):
            for p in range(21):
                for a in range(2):
                    f=φ([d+1,p+1],a)
                    lq[d,p,a]=np.sum(f*w) 
        Δ=lq-Q
        #denominator is total of state action pairs, which is 21*10*2=420
        mse=np.sum(np.square(Δ))/420
        mse_per_episode_LFA[i,game]= mse
    mse_per_lambda_LFA[i]= mse

#%% Visualizations code for MC, TD, AND LFA
#Only the Visualizations and the discussion are left!!!! 


