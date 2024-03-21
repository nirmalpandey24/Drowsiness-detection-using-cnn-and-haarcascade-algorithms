
let navbar = document.querySelector('.navbar');
document.querySelector('#menu-bar').onclick=() =>{
    navbar.classList.toggle('active');
}


// document.getElementById('start-button').addEventListener('click', function() {
//     // Code to start the camera
//     navigator.mediaDevices.getUserMedia({ video: true })
//         .then(function (stream) {
//             document.getElementById('video').srcObject = stream;
//         })
//         .catch(function (error) {
//             console.error('Error accessing camera:', error);
//         });
// });


// document.addEventListener('DOMContentLoaded', function () {
//   const startButton = document.getElementById('start-button');
//   const videoContainer = document.getElementById('video-container');
//   const video = document.getElementById('video');

//   startButton.addEventListener('click', function () {
//       // Make a request to start the camera
//       fetch('/start_detection')
//           .then(response => {
//               if (response.ok) {
//                   // If the request is successful, show the video container
//                   videoContainer.style.display = 'block';
//                   startButton.disabled = true; // Disable the button to prevent multiple clicks
//                   startVideo();
//               } else {
//                   console.error('Failed to start the camera');
//               }
//           })
//           .catch(error => console.error('Error starting the camera:', error));
//   });

//   function startVideo() {
//       // Set the video source to the streaming endpoint
//       video.src = '/video';

//       // Check if the video can play and start playing
//       video.oncanplay = function () {
//           video.play();
//       };
//   }
// });
