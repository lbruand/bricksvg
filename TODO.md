TODO
====

 * [x] keep a name of the original piece in the svg
 * [x] Do not make the same part twice by using a cache. added a use in the svg as well.
 * [x] Isometric / axonometric is not correct.
 * [x] PART_MAP should be generate from the LPART data from leocad and the brick.scad stuff
 * [ko] Experiment with different bindings of Openscad to go fast ( no fork of external process)
 * [x] Right now we have 1 png per color. It is not efficient in terms of CPU-rendering, printing in the svg and size.
       it possible to use 2 svg/mask, one to control the transparency and the other to control the shadowing and have
       control over the color and the shadowing.
 * [x] Duotone implemented. Remove original recolorize functionality with pillow.
 * [ ] build a bestiary of mermaid files to test and improve the code.
 * [ ] Right now all the pieces are rendered into the same IMG_PX x IMG_PX canvas and then cropped.
       Maybe we should be more clever. Render with a constant density relative to the size of the part.
       Maybe we should adjust depending on the complexity of the piece.


 
