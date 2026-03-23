// src/app.ts

import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import energyRoutes from "./routes/energy.routes.js";

const app = express();

app.use(helmet());
app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:5173",
    credentials: true,
  }),
);
app.use(morgan("dev"));
app.use(express.json());

// Only Energy Updates are needed for the demo
app.use("/api/energy", energyRoutes);

app.get("/health", (req, res) => {
  res
    .status(200)
    .json({ status: "OK", timestamp: new Date().toLocaleString() });
});

export default app;
