// var suggested_palletes = {
//     Name: "Eric",
//     Age = 23
//     Job: "Freelancer",
//     Skills : "JavaScript"
//   };


let original_img_link = null
let initial_colour_id = null
let final_colour_id = null

document.getElementById('input_file').addEventListener('change', function (event) {
    var image = document.getElementById('graph');
    original_img_link = URL.createObjectURL(event.target.files[0]);
    image.src = original_img_link;
    
    const target_url = "http://127.0.0.1:8000/process_image";
    try {
        const formData = new FormData();
        formData.append("file", event.target.files[0]);
    
        const response = fetch(target_url, {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                // Extract colors from the fetched data
                const colors = data.buttons.map(buttonData => buttonData.color);
                // Call function to create buttons with the extracted colors
                document.getElementById("instruction_change_from").hidden =false;
                document.getElementById("instruction_change_to").hidden =false;
                createButtons(colors);
            });;
    } catch (error) {
        console.error(error);
    }
});

// Function to handle button click
function select_initial_colour(event) {
    const clickedButton = event.target;
    const buttonContainer = document.getElementById('button-container');
    const buttons = buttonContainer.querySelectorAll('.palette-button');
    initial_colour_id = clickedButton.id
    buttons.forEach(button => {
        // Do something with each button
        button.style.border = "2px solid white";
    });

    clickedButton.style.border = "2px solid black";
}

function createButtons(colors) {
    const buttonContainer = document.getElementById('button-container');
    buttonContainer.innerHTML = ""; // Clear previous buttons (if any)
    var idx = 1;
    colors.forEach(color => {
        const button = document.createElement('button');
        button.id = idx.toString();
        idx++;
        button.style.backgroundColor = color;
        button.className = "palette-button";
        buttonContainer.appendChild(button);
        button.addEventListener('click', select_initial_colour);
        
    });
}


document.getElementById('reset').addEventListener('click', function () {
    var image = document.getElementById('graph');
    image.src = original_img_link;
    const url = 'http://127.0.0.1:8000/reset'
    fetch(url, {
        method: "GET" // default, so we can ignore
    })
});


function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
}

let colorPicker = document.querySelector("#color-picker");
colorPicker.addEventListener("change", watchColorPicker, false);

function watchColorPicker(event) {
    final_colour_id = event.target.value;
}

document.getElementById("apply").addEventListener('click', function(event){
    const image = document.getElementById("graph");
    const formData = new FormData();
    formData.append("change", initial_colour_id);


    const button = document.getElementById(initial_colour_id)
    button.style.backgroundColor = final_colour_id;

    formData.append("new_colour_hex", final_colour_id);

    // Make a POST request to http://127.0.0.1:8000/1/ with the button ID
    const response = fetch('http://127.0.0.1:8000/change/', {
        method: 'POST',
        body: formData
    })
        .then((response) => response.blob())
        .then((blob) => {
            const objectURL = URL.createObjectURL(blob);
            image.src = objectURL;
        });;
})

