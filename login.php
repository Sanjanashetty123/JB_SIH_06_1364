<?php
session_start();

// Connect to the database
$servername = "localhost";
$username = "your_db_username";
$password = "your_db_password";
$dbname = "your_db_name";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST["username"];
    $password = $_POST["password"];

    // You should hash the password before storing it in the database.
    // For this example, we assume the password is stored in plain text.
    $sql = "SELECT * FROM users WHERE username = '$username' AND password = '$password'";
    $result = $conn->query($sql);

    if ($result->num_rows == 1) {
        // Login successful
        $user = $result->fetch_assoc();
        $_SESSION["user_id"] = $user["id"];
        echo "Login successful!";
        // Redirect the user to their profile or another page here.
    } else {
        // Login failed
        echo "Login failed. Invalid username or password.";
    }
}

$conn->close();
?>