# Determining HOG parameters
How do I determine a suitable set of parameters for HOG in my context? The number of pixels_per_cell, cells_per_block, or orientations? We have visually checked and tweaked the HOG desriptor to an extent (see image below), but does this look decent? Or can we do better?


https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=6f66b0c6-6ba8-11ed-b5bd-6595d9b17862

https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=6932dcaa-6baa-11ed-b5bd-6595d9b17862

<iframe frameborder="0" class="juxtapose" width="100%" height="1353" src="https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=da1ea0bc-6c12-11ed-b5bd-6595d9b17862"></iframe>


# Shadow
I don't think there is anything I can d about shadows, is there? Since I am working with multiple images from different points in time, there is bound to be differences in camera position, and as a result, the perspective of different images (see images).


# Aligning Images + Shadows
What approach can I use to align images? I'm thinking of searching a range (+/- 25) and get the coordinate that gives me the least SSE? 
    - Will it be okay using np.rollaxis()?
    - Can I take core of shadows in any way?
Also wondering if there's a way that does this more effiently? What libraries should we check out?

# SVM on patches vs Simple Differencing