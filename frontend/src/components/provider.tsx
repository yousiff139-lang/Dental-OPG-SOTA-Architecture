"use client";
import React, { PropsWithChildren } from "react";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { Toaster } from "./ui/sonner";

const qc = new QueryClient();

const Provider = ({ children }: PropsWithChildren) => {
  return (
    <>
      <QueryClientProvider client={qc}>
        {children}
        <Toaster />
      </QueryClientProvider>
    </>
  );
};

export default Provider;

