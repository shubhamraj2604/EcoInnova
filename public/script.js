document.addEventListener("DOMContentLoaded", function () {
    const categoriesItem = document.querySelector(".categories");
    
    categoriesItem.addEventListener("click", function () {
        document.querySelector("#second").scrollIntoView({ behavior: "smooth" });
    });
});