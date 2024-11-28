const inputUploadFile = document.getElementsByClassName("button primary");
inputUploadFile.onchange = function () {
    imageUploadInput.src = URL.createObjectURL(inputUploadFile.files[0]);
}