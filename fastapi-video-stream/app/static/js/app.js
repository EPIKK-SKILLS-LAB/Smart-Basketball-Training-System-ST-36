// app/static/js/app.js
document.addEventListener("DOMContentLoaded", function() {
    const videoPlayer = document.getElementById("videoPlayer");
    const playButton = document.getElementById("playButton");
    const pauseButton = document.getElementById("pauseButton");

    playButton.addEventListener("click", function() {
        videoPlayer.play();
    });

    pauseButton.addEventListener("click", function() {
        videoPlayer.pause();
    });
});