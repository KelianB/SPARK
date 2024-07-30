function makeZoomable(selector) {
	let elements = document.querySelectorAll(selector);
	elements.forEach((element) => {
		let wrapper = document.createElement("div");
		wrapper.classList.add("zoomable-img");
		const id = `zoom${Math.round(Math.random() * 1e9)}`;
		wrapper.innerHTML = `
			<input type="checkbox" id="${id}">
			<label for="${id}"></label>
		`
		element.parentNode.replaceChild(wrapper, element);
		wrapper.querySelector("label").appendChild(element);
	});
}

$(document).ready(function() {
    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    bulmaSlider.attach();

	makeZoomable(".make-zoomable");
});
