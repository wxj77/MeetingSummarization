(Mashable) -- Call it Cloudgate, Cloudpocalyse or whatever you'd like, but the extended collapse of Amazon Elastic Cloud Compute (EC2) is both a setback for cloud computing and an opportunity for us to figure out how to stop it from happening again.

Amazon may be best-known for its online shopping site, but it also has a substantial cloud computing business. It provides a scalable, flexible and particularly efficient solution for companies to store and deliver massive amounts of content.

Its model of only paying for what you consume was a radical innovation when it launched in 2006.

In fact, Amazon Web Services has been so affordable and reliable that thousands of companies from Foursquare to Netflix utilize the company's cloud computing technology and servers to run their businesses.

They put their faith in Amazon's cloud because there was no reason to think that it would falter. One of cloud computing's key tenants is reliability through redundancy of both servers and data centers.

Then on Wednesday, Amazon's northern Virginia data center started experiencing problems that caused major latency and connectivity issues.

The trouble was apparently due to excessive re-mirroring of its Elastic Block Storage (EBS) volumes -- this essentially created countless new backups of the EBS volumes that took up Amazon's storage capacity and triggered a cascading effect that caused downtime on hundreds (or more likely thousands) of websites for almost 24 hours.

The collapse took its share of victims. Among the most prominent companies affected were Foursquare, Quora, Hootsuite, SCVNGR, Heroku, Reddit and Wildfire, though hundreds of other companies big and small were affected.

Luckily, one of Amazon's most prominent customers, Netflix, didn't experience problems because it's built for the loss of an entire data center, while companies relying on Amazon's four other global data centers didn't experience too many issues.

A learning moment

FathomDB founder Justin Santa Barbara has a detailed post on his blog about what may be the biggest problem to come out of this week's collapse: Amazon's cloud redundancies failed to stop a mass outage.

Its Availability Zones are supposed to be able to fail independently without bringing the whole system down. Instead, there was a single point of failure that shouldn't have been there.

This week's disaster in the cloud is a reminder to startups to build redundancy into their applications and their own systems, but as Santa Barbara points out, most startups don't have the time or resources to engineer for multiple cloud systems (each Amazon global region/data center has its own rules and features, making a simple "switch" to another center difficult).

These companies trusted Amazon to keep them online, and Amazon failed to deliver.

Catastrophic issues will always occur, but in the pre-cloud era, downtime only affected a single computer or website. Today, a catastrophic event takes down thousands of websites, causing millions or even billions of dollars in lost revenue and productivity.

This incident is no reason for us to shun cloud computing, though. Its benefits (scalability, cost reduction, device independence, performance and more) far outweigh its cons.

We do need to take a hard look at how we structure our cloud infrastructure though and find new ways to either prevent single points of failure or quickly move content off failing clouds faster, especially as the world's computing power is consolidated into fewer and fewer systems.

Cloud computing is still in its infancy, and today's events make it clear that we still have a lot of work to do. It could be a whole lot worse next time if we aren't prepared.

See the original article at Mashable.com.

© 2013 MASHABLE.com. All rights reserved.

@highlight

On Wednesday, Amazon Elastic Cloud Compute collapsed

@highlight

The trouble was due to excessive re-mirroring of its Elastic Block Storage volumes

@highlight

The collapse affected were Foursquare, Quora, Hootsuite, SCVNGR, and many others