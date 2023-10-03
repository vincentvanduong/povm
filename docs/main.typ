#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customise this template and discover how it works.
#show: project.with(
  title: "Note on POVM and Control",
  authors: (
    (name: "Vincent Văn Dương", email: "vv2102@nyu.edu", affiliation: "New York University"),
  ),
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!
// typst watch main.typ

= Introduction

#include("sec_intro.typ")

#include("sec_optimal_design.typ")

= Optimisation Framework

#include("sec_optimisation.typ")

= Numerical Results

#include("sec_results.typ")

= Comments

#include("sec_comments.typ")

= Conclusion

#include("sec_conclusion.typ")

#pagebreak()
#bibliography("bibliography.bib")
