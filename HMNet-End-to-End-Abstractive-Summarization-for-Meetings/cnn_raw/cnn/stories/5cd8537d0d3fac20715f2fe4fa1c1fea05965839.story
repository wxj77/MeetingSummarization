(CNET) -- If you want to consider a difficult computational problem, try thinking of the algorithms required to animate more than 10,000 helium balloons, each with its own string, but each also interdependent on the rest, which are collectively hoisting aloft a small house.

The production team at Pixar faced many new technological challenges on "Up," its tenth feature film.

That was the challenge the production team at Pixar faced when it set out to begin work on "Up," its tenth feature film, five years in the works, which hits theaters on Friday.

There was absolutely no way the team was going to hand-animate the balloons. Not with their numbers in five-figures, and especially not when you consider that within the cluster, every interaction between two balloons has a ripple effect: If one bumped another, the second would move, likely bumping a third, and so on. And every bit of this would need to be seen on screen.

In "Up," the story revolves around the main character, 78-year-old Carl Fredricksen, who, frustrated with his mundane life, ties the thousands of balloons to his house and sets off for adventures in South America. A small boy ends up marooned on board, and hilarity ensues.

The cluster of balloons is so central to the film's branding--it's called "Up," after all--that to promote the film, Pixar teamed up with two of the world's cluster ballooning experts for a nationwide tour involving a real-life flying armchair and dozens of huge, colorful balloons.

"You have a movie that's about a house that flies, which is a pretty far-fetched idea," said Steve May, the supervising technical director on "Up." "We all know, from kids' parties, how a bunch of balloons behave, so if we could animate balloons in a realistic way, the believability that the house could fly would sell."

For May, "Up" producer Jonas Rivera, director Pete Docter, and the many others involved in making the film, believability was key, even within the context of a story about a flying house. And while a major part of instilling that believability must come from a well-conceived and executed story and script, the animation is no less responsible for winning over potentially skeptical audiences.

Balloons, the mother of animation invention May said that the animation department at Pixar never even considered hand-animating the balloons. But even standard computer animation wouldn't be up to the task, because of the N-squared complexity involved in the thousands of interdependent balloons. Instead, the studio's computer whizzes figured out a way to turn the problem over to a programmed physical simulator, which, employing Newtonian physics, was able to address the animation problem.

"These are relatively simple physical equations, so you program them into the computer and therefore kind of let the computer animate things for you, using those physics," said May. "So in every frame of the animation, (the computer can) literally compute the forces acting on those balloons, (so) that they're buoyant, that their strings are attached, that wind is blowing through them. And based on those forces, we can compute how the balloon should move."

This process is known as procedural animation, and is described by an algorithm or set of equations, and is in stark contrast to what is known as key frame animation, in which the animators explicitly define the movement of an object or objects in every frame.

Procedural animation has been around for some time, but May suggested that even the most difficult uses of it in the past don't come close to what Pixar had to achieve in "Up."

Pixar fans may remember the scenes in "Cars" of a stadium full of 300,000 car "fans" cheering on a high-speed race below, each of which was independently animated. That, too, was done with procedural animation, May said, since creating so many cars individually would have been a non-starter. But even that complex computation problem didn't approach the balloon cluster issue in "Up": the "Cars" scene involved no interdependent physics.

Getting the simulator humming properly is no easy task, as one might imagine. May said it involves setting rules for how individual objects should behave, giving the computer these initial conditions, and then "let it run."

Oddly, because the simulator does indeed run with those conditions and rules and the peculiarities of physics, the animators found themselves without precise control of what would happen with the balloons--or other objects in the film animated using these techniques.

"If the (balloon cluster) is moving too slow, we increase the amount of wind, and then run the simulator again," May said. "Then maybe we turn the wind down. It's a little fun science experiment where sometimes, hopefully by the end, we're getting what we want."

Losing control of balloons Sometimes, given the vagaries of physics and chaos theory, unexpected things happen. The computer team inputs the rules and because some of the initial conditions are random, "you get semi-random results." One of May's favorite examples is that early in the film, when the house first is hoisted aloft by the balloons, a small group of the balloons actually broke off of the main cluster.

May said that this breakaway group of balloons is actually visible--albeit very briefly--in "Up." Eagle-eyed moviegoers can see the escaped balloons in the upper right-hand side of the screen, he said.

"We didn't mean for that to happen," he said, "but (we said) 'It's cool, let's keep it.'"

Even being able to make such choices wasn't possible at the beginning of the film's production, however. May said Pixar's physical simulator, an open-source program called ODE, couldn't initially handle the complexity of modeling the behavior of more than 10,000 balloons.

"We could handle about 500 (balloons), and we knew we needed tens of thousands," he said. "We knew we needed to develop a new simulator software pipeline...to handle an order of magnitude more complex simulation."

Of course, at Pixar, adjusting to evolving computer needs on the fly is nothing new. In fact, May said the studio has done so in one form or another on many of its films. For example, he said that when the studio made "Monsters, Inc.," it had to figure out how to animate the movie's monsters' fur. Similarly, when Pixar made "Finding Nemo," the animators had to figure out how to simulate underwater scenes.

"We had to learn about (how light refracts under water), and murk and how particulates float under water," May said.

And in "Up," too, there were additional animation challenges. Among them were figuring out how to animate and render the feathers on Kevin, a bird that is a major character in the film, and how to make the cloth on (main character) Carl's clothes seem believable.

Carl's threads were "the hardest clothing we've ever had to animate here," said May, "in part because Carl's a (small) man in an oversized suit. That was another case of (using) the physical simulation, and of setting up rules for how cloth should behave. And the looser the clothing, the more it can behave badly."

Even Carl himself presented some animation difficulties, May said, because the character's head is shaped like a cube.

Like many other elements in "Up," the cube-shape of Carl's face wasn't a random whim of the director. Rather, it is a story element: May explained that Carl's character is based on someone who, as a young man, was vivacious and adventurous. But as he grew older, his small house became more and more surrounded by buildings, and "it's like his world has compressed him into a square."

Thus, a cube-like face. But May said animating his facial expressions, which must fit into this cube shape, was complicated. Smiles, for example, had to come up and wrap around his cheek.

Still, for the award-winning filmmakers at Pixar, the goal is to make even the hardest animation problems look simple on the silver screen.

As producer Jonas Rivera put it, "The audience looks at (the balloon cluster) and says, 'Oh, that's pretty.' But they have no idea how much work went into it. We worked on that for over a year. (Then) the kid takes off his hat and runs his fingers through his hair. My mother will never know that took 15 people six weeks."

© 2009 CBS Interactive Inc. All rights reserved. CNET, CNET.com and the CNET logo are registered trademarks of CBS Interactive Inc. Used by permission.

@highlight

Pixar faced many new technological challenges on its film "Up," opening Friday

@highlight

The movie is about an old man who flies away on a house lifted by balloons

@highlight

Pixar used a programmed physical simulator to animate thousands of  balloons

@highlight

Studio's goal is to make even the hardest animation problems look simple on screen