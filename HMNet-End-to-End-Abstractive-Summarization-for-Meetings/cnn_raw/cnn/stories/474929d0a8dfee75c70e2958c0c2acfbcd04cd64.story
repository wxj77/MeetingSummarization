The critical Java vulnerability that is currently under attack was made possible by an incomplete patch Oracle developers issued last year to fix an earlier security bug, a researcher said.

The revelation, made Friday by Adam Gowdiak of Poland-based Security Explorations, is the latest black eye for Oracle's Java software framework which is installed on more than 1 billion PCs, smartphones, and other devices.

Miscreants use these exploits to turn compromised websites into platforms for silently installing keyloggers and other types of malicious software on the computers of unsuspecting visitors.

Last year saw a steady stream of attacks that exploited Java vulnerabilities, allowing miscreants to surreptitiously install keyloggers and other malicious software when unwitting people browsed compromised websites. The abuse has already continued into 2013, when on Thursday researchers reported yet another critical bug that is being "massively exploited in the wild".

According to Gowdiak, the latest vulnerability is a holdover from a bug (referred to here as Issue 32) that Security Explorations researchers reported to Oracle in late August. Oracle released a patch for the issue in October but it was incomplete, he said in an e-mail to Ars that was later published to the Bugtraq mailing list.

"Bugs are like mushrooms, in many cases they can be found in a close proximity to those already spotted," Gowdiak wrote. "It looks like Oracle either stopped the picking too early or they are still deep in the woods."

Oracle representatives didn't immediately respond to a request for comment. This post will be updated if a reply comes later.

People who don't use Java much should once again consider unplugging Java from their browser, while those who don't use it at all may want to uninstall it altogether. The release notes for Java 7 Update 10—the most recent version—say users can disable the program from the browser by accessing the Java Control Panel. KrebsOnSecurity has instructions here for other ways to do this.

Exploits of the latest Java vulnerability, which were first observed more than a month ago, are the combination of two bugs. The first involves the Class.forName() method and allows the loading of arbitrary (restricted) classes. The second bug relies on the invokeWithArguments method call and was also a problem with Issue 32 that Oracle purportedly patched in October.

"However, it turns out that the fix was not complete as one can still abuse invokeWithArguments method to setup calls to invokeExact method with a trusted system class as a target method caller," Gowdiak wrote. "This time the call is however done to methods of new Reflection API (from java.lang.invoke.* package), of which many rely on security checks conducted against the caller of the target method."

Developers of the Metasploit framework for hackers and penetration testers have released a module that should exploit the vulnerability on machines running Windows, Apple OS X, and Linux regardless of the browser they're using. The US-CERT, which is affiliated with the Department of Homeland Security, is advising people to disable Java in Web browsers.

@highlight

Java vulnerability is due to incomplete patch by Oracle, researcher says

@highlight

Java's software is on more than 1 billion PCs, smartphones and other computers

@highlight

People who don't use Java are encouraged to uninstall it