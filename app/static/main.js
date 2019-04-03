window.onscroll = function() {
      fixHeaderOnScroll()
};

var header = document.getElementById("main-nav-bar");
var sticky = header.offsetTop;

function fixHeaderOnScroll() {
  if (window.pageYOffset > sticky) {
    setTimeout(header.classList.add("sticky"), 500);
  } else {
    setTimeout(header.classList.remove("sticky"), 500);
  }
}
