# PDTracker_Backend

To setup the project locally, install python3 version 3.10 
and create a vitual environment using the following command:

```bash
python3.10 -m venv venv
```

Activate the virtual environment using the following command:

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Start the server using the following command:

```bash
python manage.py runserver
```

Server runs on port 8080 

Now go to VideoAnalysisToolFrontend and run `npm start` to start UI


# Running the Backend with Docker Compose

## Prerequisites
- Ensure Docker and Docker Compose are installed on your system.

## Steps

1. **Build the Docker Images:**

    ```bash
    docker compose build
    ```

    This command builds the Docker images specified in your `docker-compose.yml` file.

2. **Start the Containers:**

    ```bash
    docker compose up -d
    ```

    This command starts the containers in detached mode. The `-d` flag ensures that the containers run in the background.

3. **Stop and Remove Containers with Volumes:**

    ```bash
    docker compose down --volumes
    ```

    This command stops and removes the containers, along with any associated volumes, ensuring a clean state.

---

You can now efficiently build, start, and stop your backend using Docker Compose with these commands.
