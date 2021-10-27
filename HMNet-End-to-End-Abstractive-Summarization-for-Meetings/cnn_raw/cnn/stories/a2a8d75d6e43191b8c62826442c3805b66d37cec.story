(CNN) -- North Korea, with its previous technologically laggard image, may have just shocked the world with some alleged hacking savvy, but when ISIS comes to mind, so does the terrorists' digital bent.

The Islamist militants renowned for their bloodthirsty beheading videos and slick social media propaganda, may have extended their skills into low-level hacking, a cyber-security human rights group believes.

The Citizen Lab obtained new malware that has targeted the ISIS opposition group "Raqqa is being Slaughtered Silently," or RSS, and released an analysis of it Thursday.

The researchers from the University of Toronto can't confirm that the cyberattack is coming from the Islamic State in Iraq and Syria, especially since the Syrian regime led by Bashar al-Assad has also used Trojan horse software to fight activists since 2011.

But the workings of the malware, its intended target and what it achieves for the attacker lead The Citizen Lab to suspect ISIS the most.

ISIS hates RSS

ISIS is particularly motivated to strike RSS.

The Islamist extremist militants like to depict their stronghold city of Raqqa to the world as a caliphate paradise, where life under strictest Sharia is practically Heaven on Earth.

But RSS activists in the city reveal on social media Raqqa's bleeding underbelly, the terrorizing of residents.

Warnings of graphic content speckle its Twitter feed, where photos of public beheadings and stonings of residents in Syrian cities are posted in unflinching detail.

RSS also reports coalition airstrike hits against ISIS and warns Raqqa residents about new strict Sharia rules the militants impose on them.

But activists participating in RSS activities have another enemy. Before ISIS took over their town, they were taking the same actions against the Syrian regime.

Slapdash malware

The Raqqa target who passed along the malware to The Citizen Lab did not fall for its ploy, and the group was not successfully hacked, as far as The Citizen Lab researcher John Scott-Railton knows.

But he fears others who may have received the target email may not have been so savvy or lucky.

The malware used in the attack is simple and lean, and whoever wrote it did some things wrong -- or felt it wasn't necessary to do them right.

The Citizen Lab found the malware to be effective and very dangerous even without proper coding whistles and bells, because its targeting of victims is socially savvy.

This is how it works.

The victim receives an enticing email tailored to his anti-ISIS interests from people claiming to be expat Syrian activists living in Canada. They ask for the local activist's help in working with mainstream media.

"We are preparing a lengthy news report on the realities of life in Raqqah," the email reads. "We are sharing some information with you with the hope that you will correct it in case it contains errors."

Images are attached showing areal photos with spots marked on them portraying alleged ISIS strongholds and U.S. airstrike targets

And the email includes a link to a file sharing site, where the victim is encouraged to download files, which contain a slideshow of more such images.

But in the download is also a malicious file, and while the victim views more photos, it installs a set of small malware files onto the target's computer.

Find them, punish them

Once there, these bad files don't do much, The Citizen Lab said. Just enough.

"The custom malware ... beacons home with the IP address of the victim's computer and details about his or her system each time the computer restarts," it said in its study.

That's enough for militants who know the area to determine the user's physical location.

The files don't include a key logger -- although many forms of the software that monitors what infected users are writing are readily available on the Internet.

Such RATs (Remote Access Trojans) are typical of the Syrian regime, whose hackers seems more interested in obtaining opposition activist content, the researchers Scott-Railton and Seth Hardy said.

"A RAT would have provided much greater access alongside IP information," they said.

It makes the researchers think that whoever is using the slideshow malware may be interested only in "identifying and locating a target."

Find his Internet café or apartment; haul him in; punish him -- or execute him. That's probably the idea, The Citizen Lab said.

American journalist James Foley, who ISIS later beheaded, was captured coming out of an Internet café in Syria in 2012 before ISIS officially existed.

Regime's signature different

This attempted hack doesn't bear the signature of typical Syrian regime attackers, The Citizen Lab said.

They usually employ servers to facilitate data sent back by their RATs, but this malware doesn't need one. It sends an attachment with the sparse information it gathers to an email account.

"This functionality would be especially useful to an adversary unsure of whether it can maintain uninterrupted Internet connectivity," the researchers said.

Whether shoddiness or simplicity: That email is improperly encrypted, leaving the recipient's logon credentials open to interception. One of the malware's passwords is also visible in its code.

There are other apparent bugs, and software itself is unusually artless.

"It relies on a half dozen separate executable files, each with a single task," the researchers said.

But keeping it bare bones has an advantage, the researchers said.

"The program looks less like malware, and may attract less attention from endpoint protection tools and scanners. Detections were low when the file was first submitted to VirusTotal, for example. It registered only 6/55 detections by anti-virus scanners, or a 10% detection rate."

This malware flies under the radar.

-

@highlight

Target email speaks to victims who are opposed to ISIS and asks for their help

@highlight

It contains a link to a file sharing site, where a malicious file is hidden among photos

@highlight

Malware is artless, and the writer encrypted it wrong

@highlight

But it's dangerous: Being bare-bones makes it hard for security software to detect