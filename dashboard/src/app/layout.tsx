import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenMausBot + 9Router | Observability Dashboard",
  description: "Real-time token usage, 9Router routing success rate, and task queue monitor",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
