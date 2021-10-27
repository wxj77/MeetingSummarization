(WIRED)  -- Scofflaws could hack the smart cards that access electronic parking meters in large cities around the United States, researchers are finding.

"Cities all over the nation and all over the world are deploying these smartcard meters," researcher says.

The smart cards pay for parking spots, and their programming could be easily changed to obtain unlimited free parking.

It took researcher Joe Grand only three days to design an attack on the smart cards. The researchers examined the meters used in San Francisco, California, but the same and similar electronic meters are being installed in cities around the world.

"It wasn't technically complicated and the fact that I can do it in three days means that other people are probably already doing it and probably taking advantage of it," said Joe Grand, a designer and hardware hacker and one of the hosts of the Discovery Channel's "Prototype This" show. "It seems like the system wasn't analyzed at all."

Grand and fellow researcher Jake Appelbaum presented their findings Thursday afternoon at the Black Hat security conference in Las Vegas, Nevada. The researchers did not contact the San Francisco Municipal Transportation Agency or the meter maker prior to their talk, and asked reporters not to contact those organizations ahead of their presentation, for fear of being gagged by a court order.

At last year's DefCon hacker conference, MIT students were barred from talking about similar vulnerabilities in smartcards used by the Massachusetts Bay Transportation Authority after the MBTA obtained a restraining order. They spoke with Threat Level about their findings prior to the presentation.

"We're not picking on San Francisco," Grand said. "We're not even claiming to get free parking. We're trying to educate people about ... how they can take our research and apply it to their own cities if they are trying to deploy their own systems or make them more secure.... Cities all over the nation and all over the world are deploying these smartcard meters [and] there's a number of previously known problems with various parking meters in other cities."

San Francisco launched a $35-million pilot project in 2003 to deploy smart meters around the city in an effort to thwart thieves, including parking control officers who were skimming money from the meters.

The city estimated it was losing more than $3 million annually to theft. In response, it installed 23,000 meters made by a Canadian firm named J.J. MacKay, which also has meters in Florida, Massachusetts, New York, Canada, Hong Kong and other locales.

The machines are hybrids that allow drivers to insert either coins, or a pre-paid GemPlus smart card, which can be purchased in values of $20 or $50. The machines also have an audit log to help catch insiders who might skim proceeds.

To record the communication between the card and the meter, Grand purchased a smartcard shim -- an electrical connector that duplicates a smartcard's contact points -- and used an oscilloscope to record the electrical signals as the card and meter communicated.

He discovered the cards aren't digitally signed, and the only authentication between the meter and card is a password sent from the former to the latter. The card doesn't have to know the password, however, it just has to respond that the password is correct.

The cards sold in San Francisco are designed to be thrown out when the customer has exhausted them. But the researchers found that the meters perform no upper-bounds check, so hackers could easily boost the transaction limit on a card beyond what could legitimately purchased. They could also program a card to simply never deduct from the transaction count.

"We're residents of San Francisco and our taxes are going towards a broken system that they could potentially be losing money on and we pay the consequences of that," Grand said.

Subscribe to WIRED magazine for less than $1 an issue and get a FREE GIFT! Click here!

Copyright 2009 Wired.com.

@highlight

Researchers find smart cards that access parking meters can be hacked

@highlight

Joe Grand and Jake Appelbaum present findings at Black Hat conference

@highlight

Before smart cards, city was losing more than $3 million annually to theft

@highlight

Researchers programmed cards to never deduct from transaction count