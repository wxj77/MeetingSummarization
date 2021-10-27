Editor's note: Tomorrow Transformed explores innovative approaches and opportunities available in business and society through technology.

(WIRED) -- The design studio Nervous System has created a novel process that allows a 3-D printed dress to move and sway like real fabric. The bespoke software behind it, called Kinematics, combines origami techniques with novel approaches to 3-D printing, pushing the technology's limits.

Instead of pinning fabric to a dress form, a Kinematics garment starts as a 3-D model in a CAD program. Kinematics breaks the model down into tessellated, triangular segments of varying sizes. Designers can control the size, placement, and quantity of the triangles in a Javascript-based design tool and preview how the changes will impact the polygonal pinafore. Once the designer is satisfied, algorithms add hinges to the triangles uniting the garment into a single piece and compress the design into the smallest possible shape to optimise the printing process, often reducing the volume by 85 percent.

After two days of printing at Shapeways, a dusty boulder of plastic emerges from an industrial-sized 3-D printer. Technicians remove excess dust like archeologists in search of a long-buried garment. The plastic parts are cleaned and dyed, resulting in a little black (or white) dress made from tiny, interlocking bricks of plastic.

No Gimmicks for this Gown

Designer Jessica Rosenkrantz made sure the gown was more than mere gimmickry. Buttons, cleverly modelled into the triangles make it easy to don and doff. Unlike other 3-D printed clothing that feels like a suit of armour, the long dress flows and moves as the model strides and twirls.

What Cities Would Look Like if Lit Only by the Stars

Comfort was a key concern. Rosenkrantz wore 3-D printed jewellery for weeks at a time in an attempt to catch design features that chafe. She built her wardrobe piece by piece, starting with a bracelet, then a belt, and finally a bodice before moving on to a dress. Rosenkrantz brought an old-school tailor's approach to the project, but was happy to leverage modern technology. For example, 3-D scans of the model's body ensured a perfect fit. She worked with Shapeways to optimise the print quality and aesthetics. As a result, her garment and its Github repository recently were acquired by the Museum of Modern Art.

Making It Work

Nervous System originally developed the Kinematics concept as part of a project for Google. The goal was to help add bit of cool to a pavilion promoting Android phones. Nervous System figured out how to print bracelets on MakerBots by reducing dimensional designs to flat pieces of plastic that could be printed in under an hour and folded like origami. Google was pleased with the promotion, but Nervous System believed the concept could be used to make garments. "We'd done some simulations and made some animations showing that we could do it hypothetically," says Rosenkrantz.

These hypothetical simulations precipitated a software engineering effort one year in the making. Scaling up from a wrist-worn wearables to cocktail dress posed a particular challenge. The hinges linking the triangles must be small enough to let the fabric flow, but robust enough to avoid a wardrobe malfunction.

These mechanical challenges were exacerbated by limitations in 3-D printing technology. Pieces made with the technology have a grain, like wood, and certain orientations create stronger parts. The solution was to revamp the software. "We were able to do so much design-wise without ever printing anything," says Rosenkrantz. "We knew not only exactly what the final piece would look like but also how it would behave." Simulating folds was slow and inaccurate at first. Test prints of belts with 77 hinges worked beautifully, but scaling up to the 700 or more needed to create a dress repeatedly broke the software. Physics engines were tossed aside like fabric swatches.

15 Incredible Photos That'll Remind You to Be Awed by Planet Earth

Originally, the simulator would fold the clothes down into a ball. "Sort of like you are wadding clothes up to toss in you hamper," says Rosenkrantz. "It looked cool but it wasn't the most efficient way to get the volume of our designs down." So Rosenkrantz and partner Jesse Louis-Rosenberg developed a collision-based simulator that replicates how one might fold clothes to put them in a drawer.

The project pushed design, fashion, and fabrication in surprising ways. "To 3-D print structures in this crazy compressed form and have them unfold; that almost sounds like science fiction," says Rosenkrantz. "Frankly, when you work on something complex like this in a completely digital world for so long, the biggest surprise is that it actually works as intended, from the compressing to the fit, draping, and movement."

Printing also required special development. Nervous System needed to develop new tools to load its software. "We've been working with Nervous and our community over the years to push the machines to their limits," says Carine Carmy of Shapeways. "From how densely we can pack the trays so you can print 1,000 products at once versus just one, to how long you need to run them so we can produce products more quickly, to how precise and detailed the prints can be so that you can design with micron precision."

Ready to Wear?

Next up for Nervous System is improving the speed and adding new mechanisms and structures that will allow simulating different materials -- think of a stout tweed versus a gossamer silk. Ultimately, the team thinks can be expanded for other applications like Skylar Tibbits Hyperform project.

At $3,000 a pop, Nervous System isn't quite ready to commercialise its wearable wares. "That is a very high number although perhaps considerably lower than the price of most other 3-D printed garments," she says. "We're hoping to bring the price down before we start selling clothing."

Read more from WIRED:

21 Awesomely Well-Designed Products We're Dying to Own

The Ruins of the USSR's Secret Nuclear Cities

What Would Your Ideal, Photoshopped Face Look Like? 14 People Find Out

Subscribe to WIRED magazine for less than $1 an issue and get a FREE GIFT! Click here!

Copyright 2011 Wired.com.

@highlight

A design studio has created a 3D printed dress that sways like real fabric

@highlight

The software behind the process combines origami techniques with 3D printing

@highlight

The company originally developed the concept as part of a Google project

@highlight

At $3,000 a pop, it isn't quite ready for the retail market yet