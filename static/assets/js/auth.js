const auth = firebase.auth();
var credentials;
var resolvedCredentials = false;
var windowLoaded = false;
auth.onAuthStateChanged(function(user) {
	credentials = user;
	if(credentials == null) {
		if(window.location.pathname.startsWith("/record") || window.location.pathname.startsWith("/add")) {
			window.location.href = "/login";
		} else if(window.location.pathname.startsWith("/login") || window.location.pathname.startsWith("/register")) {
			window.location.href = "/records";
		} else {
			resolvedCredentials = true;
		}
	} else {
		document.getElementById("nav-sign-in").remove();
		document.getElementById("nav-sign-up").innerHTML = '<a href="javascript:auth.signOut();">Sign Out</a>';
		resolvedCredentials = true;
	}
	if(windowLoaded && resolvedCredentials) {
		if($('.cover').length){
			$('.cover').parallax({
				imageSrc: $('.cover').data('image'),
				zIndex: '1'
			});
		}

		$("#preloader").animate({
			'opacity': '0'
		}, 600, function(){
			setTimeout(function(){
				$("#preloader").css("visibility", "hidden").fadeOut();
			}, 300);
		});
	}
});
$(window).on('load', function () {
	windowLoaded = true;
	if(windowLoaded && resolvedCredentials) {
		if($('.cover').length){
			$('.cover').parallax({
				imageSrc: $('.cover').data('image'),
				zIndex: '1'
			});
		}

		$("#preloader").animate({
			'opacity': '0'
		}, 600, function(){
			setTimeout(function(){
				$("#preloader").css("visibility", "hidden").fadeOut();
			}, 300);
		});
	}
});