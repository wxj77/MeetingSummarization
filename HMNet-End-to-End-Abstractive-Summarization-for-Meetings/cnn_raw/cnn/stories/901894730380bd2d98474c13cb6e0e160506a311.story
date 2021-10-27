(CNET)  -- When Facebook Chief Executive Mark Zuckerberg recently announced a "Like" button that publishers could place on their Web pages, he predicted it would make the Web smarter and "more social."

What Zuckerberg didn't point out is that widespread use of the Like button allows Facebook to track people as they switch from CNN.com to Yelp.com to ESPN.com, all of which are sites that have said they will implement the feature.

Even if someone is not a Facebook user or is not logged in, Facebook's social plugins collect the address of the Web page being visited and the Internet address of the visitor as soon as the page is loaded -- clicking on the Like button is not required.

If enough sites participate, that permits Facebook to assemble a vast amount of data about Internet users' browsing habits.

"If you put a Like button on your site, you're potentially selling out your users' privacy even if they never press that button," says Nicole Ozer, an attorney with the ACLU of Northern California. "It's another example of why user control needs to be the default in Facebook."

In the last few months, scrutiny of the privacy practices of the Internet's second most popular Web site has reached an all-time high, with politicians threatening probes and privacy activists calling for formal investigations.

In response to the outcry, Zuckerberg convened a press conference last week at Facebook's Palo Alto, California, headquarters, where he pledged to make privacy "simpler."

For its part, Facebook told CNET on Tuesday that the information about who viewed what pages with a Like button is anonymized after three months and is not shared with or sold to third parties. A representative acknowledged, however, that the current privacy description of Facebook's social plugins "is not as clear as it could be, and we'll fix that."

Facebook's FAQ says: "No data is shared about you when you see a social plugin on an external website." No mention of this data-sharing appears under the "Information from other websites" section of the company's general privacy policy.

Publishers like "Like"

Almost as soon as Zuckerberg had finished describing the Like buttons at the F8 developer conference in April, they became a hit with Web publishers hoping for a traffic boost.

Wired's Webmonkey.com published a tutorial, a WordPress adaptation appeared, and Foursquare quickly incorporated the concept too.

Facebook itself confirmed that after only a week, "more than 50,000 sites across the Web have implemented" social plugins.

SearchEngineLand.com said Like buttons are "recommended" for virtually all Web sites; one blogging how-to guide reported that "small, blue Like buttons are now multiplying across the Web faster than you can say 'pandemic.'"

Marc Rotenberg, director of the Electronic Privacy Information Center, said that if his group had been aware of how the Like button was implemented, it would have raised this topic in a request for a Federal Trade Commission investigation of Facebook's privacy practices. (The statement sent to the FTC says, in part, that social plugins "violate user expectations and reveal user information without the user's consent.")

"The recent Facebook changes are too complex and too subtle for most users to meaningfully evaluate," Rotenberg said. "And it's not obvious that the recent announcement from Facebook has addressed all of these problems."

On the other hand, some of the Like button's features can work only if Facebook receives the user ID and URL of the Web page being visited. That allows a custom bit of Javascript code to customize the Like button.

Social plugins "work the same basic way all widgets across the Internet do," said Barry Schnitt, a Facebook spokesman. "The URL of the Web page the user is viewing must be sent to Facebook for Facebook to know where to render the personalized content."

Schnitt said Facebook does not correlate pages viewed with advertising, so someone who spends a lot of time reading articles about German sports cars on caranddriver.com will not receive Porsche 911 or Mercedes C63 AMG ads on Facebook.com. "Of course, if the user actively 'likes' that page, then it is added to their profile and they might see a related ad on Facebook," he said.

"We use the information to help improve the service," Schnitt said. "We need to see how many people see a certain Like button to know what the click-through ratio for that button is, for example. If something has a really low rate, maybe something is wrong with the site, the implementation, or our product. If it is really high, maybe something fishy is going on."

The way Facebook has implemented its Like button resembles an advertising network: Code on Facebook's systems is executed whenever someone loads a page on, say, Mashable.com, one of the Web sites that quickly adopted the button.

And advertising networks have come under significant regulatory scrutiny before, in part because they have the ability to create dossiers on what Internet users are doing across thousands or millions of different Web sites.

Ozer, the ACLU attorney, said she would caution sites to be careful before adopting Like buttons: "If an organization puts a Like button on their site, they're potentially telling Facebook about everyone who visits their Web site, every time that person visits their Web site."

How it works

Facebook wants publishers to insert an iframe or JavaScript in the HTML for their Web pages.

As soon as the page is loaded, the code invokes a PHP script at Facebook.com that records information including the URL for the Web page, your IP address, and your Facebook ID (if you're authenticated).

If a publisher uses Facebook's Javascript API, the simpler option, here's what the embedded Like button for CNET.com would look like: <fb:like href="cnet.com" font="tahoma"></fb:like>

© 2010 CBS Interactive Inc. All rights reserved. CNET, CNET.com and the CNET logo are registered trademarks of CBS Interactive Inc. Used by permission.

@highlight

Like button allows Facebook to track people as they switch websites

@highlight

Facebook's social plugins collect the address of the Web page being visited

@highlight

More than 50,000 sites across the Web have implemented social plugins