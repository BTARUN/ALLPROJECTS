How to Run a React Project on Your System

1.  Install Node.js
    -   Download from https://nodejs.org (LTS version)
    -   Verify installation: node -v npm -v
2.  Open the Project Folder
    -   Open VS Code → File → Open Folder
    -   Or via terminal: cd your-react-project-folder
3.  Install Dependencies
    -   Run: npm install
4.  Start the Development Server
    -   Run: npm start OR npm run dev
5.  Open Localhost
    -   Go to: http://localhost:3000 or as shown in your terminal.

Optional: Create a New React Project

Using Vite: npm create vite@latest cd project-name npm install npm run
dev

Using Create React App: npx create-react-app myapp cd myapp npm start

Common Errors:

-   ‘npm is not recognized’: Install Node.js again.

-   Port already in use: npm start –port=3001

-   Missing dependencies: npm install
