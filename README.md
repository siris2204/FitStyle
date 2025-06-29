# FitStyle
AI powered fashion stylist for enhanced body confidence

In recent years, online fashion platforms have experienced significant growth, transforming the 
way consumers discover and purchase clothing. While many existing fashion recommendation 
systems emphasize style compatibility, co-purchase history, or visual similarity, they often 
overlook a fundamental aspect of clothing selection: the fit based on individual body shape. This 
gap leads to suboptimal recommendations, especially for users whose body types do not conform 
to conventional standards. To address this limitation, this project presents a body-shape-aware 
fashion recommendation system that tailors outfit suggestions based on the unique measurements 
of each user. 
The system enables users to upload a full-body image directly through a Google Colab notebook. 
Using OpenPose, a powerful pose estimation tool, the system detects 2D keypoints of the body 
and extracts essential body measurements such as bust, waist, and hip circumferences. These 
measurements are then used to classify the user's body type (e.g., pear, apple, hourglass), which 
plays a pivotal role in how garments fit and appear. With the identified body shape, a K-Nearest 
Neighbors (KNN) algorithm queries a curated dataset of outfits to recommend clothing items 
that are most suitable for the user's body proportions. 
By shifting the focus from purely visual style matching to measurement-driven recommendation, 
the project aims to improve both the accuracy of fit and user satisfaction. This approach not only 
enhances the practicality of virtual try-on systems but also promotes inclusivity by 
accommodating a broader range of body types. The system is lightweight and can operate in real
time, making it accessible to users without the need for advanced hardware or extensive setup. 
Future improvements to this system include replacing OpenPose with SPIN, which offers 3D 
body mesh estimation for even greater accuracy. Additional enhancements such as style and 
color filters, a user feedback loop for active learning, and deployment as a mobile or web 
application are also planned. These will further expand the usability, flexibility, and 
personalization capabilities of the platform, moving it closer to real-world applicability in online 
fashion retail.
