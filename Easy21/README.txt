RL Game Setup

First impression

Infinte deck of cards: Randint(1,10). + How do I input the card color probabilities? In this game, black= positive, red=negative 
cardno variable to set the color probability 

Do we need a 'Game' Class t? Do we need to create a memory for SARS'A values? Anyhow, each turn's results must be saved somewhere...

We need a function called Player which launches the card drawing function inside and implements a strategy (RL algo goes here?).
This strategy must have an output that either terminates the Player's turn or recalls the card drawing function and reruns the strategy.
Always keeping the number of the score! 

This class needs to have as output the player's card sum so that the dealer's turn can begin. We need good order flow!
The dealer's strategy does not implement RL as it is rather deterministic 

