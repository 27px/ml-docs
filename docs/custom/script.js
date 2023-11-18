
_ = q => document.querySelector(q);
$ = q => Array.from(document.querySelectorAll(q));

function remove_numbering_in_nav() {
    $("nav span, head title, article h1").forEach(element => {
        const arr = element.innerText.split(". ")
        if(arr.length < 2) {
            return
        }
        element.innerText = arr.slice(1).join(". ")
    })
}

window.onload = () => {
    remove_numbering_in_nav()
}
